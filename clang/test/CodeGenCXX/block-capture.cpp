// RUN: %clang_cc1 -triple arm64e-apple-ios -x c++ -std=c++20 -fblocks -Wimplicit-block-var-alloc -fexceptions \
// RUN:            -verify -emit-llvm %s -o - | \
// RUN:    FileCheck --implicit-check-not "call{{.*}}_ZN3BigD" %s

// CHECK: %struct.__block_byref_baz = type { ptr, ptr, i32, i32, i32 }
// CHECK: [[baz:%[0-9a-z_]*]] = alloca %struct.__block_byref_baz
// CHECK: [[bazref:%[0-9a-z_\.]*]] = getelementptr inbounds nuw %struct.__block_byref_baz, ptr [[baz]], i32 0, i32 1
// CHECK: store ptr [[baz]], ptr [[bazref]]
// CHECK: call void @_Block_object_dispose(ptr [[baz]]

int a() {
  __block int baz = [&]() { return 0; }();
  (void)^{
    (void)baz;
  };
  return 0;
}

class Big {
public:
  Big(const Big &);
  ~Big();

private:
  Big();
  int s[100];
};
Big getBig(Big * (^)());

void myCleanup(void *);

// CHECK: define void @_Z11heapInitBigv
// CHECK: store i8 0, ptr {{.*}}initialized
// CHECK: call void @_Block_object_assign(ptr {{.*}}, ptr {{.*}}, i32 8)
// CHECK: call void @_Block_object_dispose(ptr {{.*}}, i32 8)
// CHECK: %B11 = getelementptr inbounds %struct.__block_byref_B1, ptr %{{.*}}, i32 0, i32 
// CHECK: invoke void @_Z6getBigU13block_pointerFP3BigvE({{[^,]*}} %B11,
//
// CHECK: invoke.cont:
// CHECK: store i1 true, ptr {{.*}}initialized
// CHECK: call void @_Block_object_dispose
// (no call to destructor, enforced by --implicit-check-not)
//
// CHECK: define internal void @__Block_byref_object_copy_
// (no call to copy constructor, enforced by --implicit-check-not)
//
// CHECK: define internal void @__Block_byref_object_dispose_
// CHECK: load {{.*}}initialized
// CHECK: br {{.*}}label %skip
// CHECK: call {{.*}} @_ZN3BigD1Ev(
// CHECK: skip:
//
void heapInitBig() {
  // expected-warning@+1{{variable 'B1' will be initialized in a heap allocation}}
  __block Big B1 = ({ // Make sure this works with statement expressions.
    getBig(
        // expected-note@+1{{because it is captured by a block used in its own initializer}}
        ^{
          return &B1;
        });
  });
}

struct Small {
  int x[2];
};
extern Small getSmall(const void * (^)());
extern Small getSmall(const void * (^)(), const void * (^)());
extern Small getSmall(__SIZE_TYPE__);
extern int getInt(const void *(^)());

// CHECK: define {{.*}} @_Z19heapInitSmallAtomicv
// CHECK: %S11 = getelementptr inbounds %struct.__block_byref_S1, ptr %{{.*}}, i32 0, i32
// CHECK: [[call:%[0-9a-z_\.]*]] = invoke i64 @_Z8getSmallU13block_pointerFPKvvE(
// CHECK: [[dive:%[0-9a-z_\.]*]] = getelementptr inbounds %struct.Small, ptr %S11, i32 0, i32 0
// CHECK: store i64 [[call]], ptr [[dive]], align 8

void heapInitSmallAtomic() {
  // expected-warning@+1{{variable 'S1' will be initialized in a heap allocation}}
  __block _Atomic(Small) S1 = getSmall(
      // expected-note@+1{{because it is captured by a block used in its own initializer}}
      ^const void * {
        return &S1;
      });
}

// With multiple blocks capturing the same variable, we only note the first one.
// In theory it would be more helpful to note each block, but that would mess up
// the grammar of the diagnostic.
void heapInitTwoSmall() {
  // expected-warning@+1{{variable 'S2' will be initialized in a heap allocation}}
  __block Small S2 = getSmall(
      // expected-note@+1{{because it is captured by a block used in its own initializer}}
      ^const void * {
        return &S2;
      },
      ^const void * {
        return &S2;
      });
}


// Test __attribute__((cleanup)) with direct init on heap.  We expect this to
// first call the cleanup function on the heap allocation, then call
// _Block_object_dispose (which runs the destructor and frees the allocation).
//
// CHECK: define {{.*}} @_Z19heapInitWithCleanupv
// CHECK: store i1 true, ptr {{.*}}initialized
// CHECK: [[fwd:%.*]] = load ptr, ptr %forwarding
// CHECK: [[obj:%.*]] = getelementptr inbounds %struct.__block_byref_B2, ptr [[fwd]], i32 0, i32 8
// CHECK: invoke void @_Z9myCleanupPv({{.*}}[[obj]])
// CHECK-NOT: ret
// CHECK: call void @_Block_object_dispose(ptr %
void heapInitWithCleanup() {
  // expected-warning@+1{{variable 'B2' will be initialized in a heap allocation}}
  __attribute__((cleanup(myCleanup))) __block Big B2 = getBig(
      // expected-note@+1{{because it is captured by a block used in its own initializer}}
      ^{
        return &B2;
      });
}

// Same as above but without direct init on heap; therefore, the stack copy
// needs to be destroyed in addition to the above operations.  This must happen
// after the cleanup call (and also happens after _Block_object_dispose, but
// doesn't need to).
//
// CHECK: define {{.*}} @_Z22nonHeapInitWithCleanupv()
// CHECK: [[fwd:%.*]] = load ptr, ptr %forwarding
// CHECK: [[obj:%.*]] = getelementptr inbounds %struct.__block_byref_B3, ptr [[fwd]], i32 0, i32
// CHECK: invoke void @_Z9myCleanupPv({{.*}}[[obj]])
//
// CHECK: invoke.cont:
// CHECK: call void @_Block_object_dispose
// CHECK: call {{.*}} @_ZN3BigD1Ev(
//
// CHECK: lpad:
// CHECK: call void @_Block_object_dispose
// CHECK: call {{.*}} @_ZN3BigD1Ev(
//
// CHECK: define internal void @__Block_byref_object_dispose_
// CHECK: call {{.*}} @_ZN3BigD1Ev(
void nonHeapInitWithCleanup() {
  __attribute__((cleanup(myCleanup))) __block Big B3 = getBig(nullptr);
  (void)^{ return &B3; };
}

// This used to cause an ICE because the type of the variable makes it eligible
// for constant initialization (but the actual initializer doesn't).
//
// It's also not actually a self-reference, so we should not get the warning.
// The block is not capturing itself; it's only capturing a pointer to itself
// (thus, e.g., calling Block_copy would not make it safe to use after the
// function returns).  You might expect C++ lifetime extension to affect this
// but it doesn't.
void constRef() {
  __block const Small &S = getSmall(^const void * {
    return &S;
  });
}

// This is also not actually a self-reference because the block literal
// is in an unevaluated expression.  We should not get the warning.
void unevaluated() {
  __block Small S = getSmall(sizeof(^const void * {
    return &S;
  }));
}

// These are not self-references because the initializers are
// const-evaluated and the block literal happens to never get executed
// (despite not being in an "unevaluated expression" type-system-wise).
void constInits() {
  __block constexpr const int I = true ? 1000 : getInt(^const void *{
    return &I;
  });
  __block constexpr Small S = true ? Small{2000, 3000} : getSmall(^const void *{
    return &S;
  });
  __block const Small &S2 = true ? Small{4000, 5000} : getSmall(^const void *{
    return &S2;
  });
}

// This is definitely not a self-reference.
void irrelevantVariable() {
  __block Small Irrelevant;
  __block Small NotCaptured = getSmall(^const void * {
    return &Irrelevant;
  });
}
