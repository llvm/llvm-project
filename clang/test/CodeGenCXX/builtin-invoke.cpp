// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

extern "C" void* memcpy(void*, const void*, decltype(sizeof(int)));
void func();

namespace std {
  template <class T>
  class reference_wrapper {
    T* ptr;

  public:
    T& get() { return *ptr; }
  };
} // namespace std

struct Callable {
  void operator()() {}

  void func();
};

extern "C" void call1() {
  __builtin_invoke(func);
  __builtin_invoke(Callable{});
  __builtin_invoke(memcpy, nullptr, nullptr, 0);

  // CHECK:      define dso_local void @call1
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ref.tmp = alloca %struct.Callable, align 1
  // CHECK-NEXT:   call void @_Z4funcv()
  // CHECK-NEXT:   call void @_ZN8CallableclEv(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp)
  // CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 null, ptr align 1 null, i64 0, i1 false)
  // CHECK-NEXT:   ret void
}

extern "C" void call_memptr(std::reference_wrapper<Callable> wrapper) {
  __builtin_invoke(&Callable::func, wrapper);

  // CHECK:      define dso_local void @call_memptr
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %wrapper = alloca %"class.std::reference_wrapper", align 8
  // CHECK-NEXT:   %coerce.dive = getelementptr inbounds nuw %"class.std::reference_wrapper", ptr %wrapper, i32 0, i32 0
  // CHECK-NEXT:   store ptr %wrapper.coerce, ptr %coerce.dive, align 8
  // CHECK-NEXT:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_ZNSt17reference_wrapperI8CallableE3getEv(ptr noundef nonnull align 8 dereferenceable(8) %wrapper)
  // CHECK-NEXT:   %0 = getelementptr inbounds i8, ptr %call, i64 0
  // CHECK-NEXT:   br i1 false, label %memptr.virtual, label %memptr.nonvirtual
  // CHECK-EMPTY:
  // CHECK-NEXT: memptr.virtual:
  // CHECK-NEXT:   %vtable = load ptr, ptr %0, align 8
  // CHECK-NEXT:   %1 = getelementptr i8, ptr %vtable, i64 sub (i64 ptrtoint (ptr @_ZN8Callable4funcEv to i64), i64 1), !nosanitize !2
  // CHECK-NEXT:   %memptr.virtualfn = load ptr, ptr %1, align 8, !nosanitize !2
  // CHECK-NEXT:   br label %memptr.end
  // CHECK-EMPTY:
  // CHECK-NEXT: memptr.nonvirtual:
  // CHECK-NEXT:   br label %memptr.end
  // CHECK-EMPTY:
  // CHECK-NEXT: memptr.end:
  // CHECK-NEXT:   %2 = phi ptr [ %memptr.virtualfn, %memptr.virtual ], [ inttoptr (i64 ptrtoint (ptr @_ZN8Callable4funcEv to i64) to ptr), %memptr.nonvirtual ]
  // CHECK-NEXT:   call void %2(ptr noundef nonnull align 1 dereferenceable(1) %0)
  // CHECK-NEXT:   ret void
}
