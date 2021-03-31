// RUN: %clang_cc1 %s -emit-llvm -triple i686-windows-msvc -o - | FileCheck %s

// Statement allow the user to exit the evaluation scope of a CallExpr without
// executing the call. Check that clang generates reasonable IR for that case.

// Not trivially copyable, subject to inalloca.
struct Foo {
  int x;
  Foo();
  ~Foo();
};

void inalloca(Foo x, Foo y);

// PR25102: In this case, clang attempts to clean up unreachable blocks *during*
// IR generation. inalloca defers some RAUW operations to the end of codegen,
// and those references would become stale when the unreachable call to
// 'inalloca' got deleted.
extern "C" void pr25102() {
  inalloca(Foo(), ({
             goto out;
             Foo();
           }));
out:;
}

// CHECK-LABEL: define dso_local void @pr25102()
// CHECK: br label %out
// CHECK: out:
// CHECK: ret void

bool cond();
extern "C" void seqAbort() {
  inalloca(Foo(), ({
             if (cond())
               goto out;
             Foo();
           }));
out:;
}

// FIXME: This can cause a stack leak. We should really have a "normal" cleanup
// that goto branches through.
// CHECK-LABEL: define dso_local void @seqAbort()
// CHECK: alloca inalloca <{ %struct.Foo, %struct.Foo }>
// CHECK: call zeroext i1 @"?cond@@YA_NXZ"()
// CHECK: br i1
// CHECK: br label %out
// CHECK: call void @"?inalloca@@YAXUFoo@@0@Z"(<{ %struct.Foo, %struct.Foo }>* inalloca(<{ %struct.Foo, %struct.Foo }>) %{{.*}})
// CHECK: call void @llvm.stackrestore(i8* %inalloca.save)
// CHECK: out:
