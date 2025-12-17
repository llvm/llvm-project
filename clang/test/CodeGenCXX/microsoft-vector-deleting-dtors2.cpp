// RUN: %clang_cc1 -emit-llvm -fms-extensions -fms-compatibility -fno-dllexport-inlines %s -triple=x86_64-pc-windows-msvc -o - | FileCheck --check-prefixes=X64,CHECK %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions -fms-compatibility -fno-dllexport-inlines %s -triple=i386-pc-windows-msvc -o - | FileCheck --check-prefixes=X86,CHECK %s

// Check that vector deleting destructor does not reference undefined symbols.
// Check that when there is no suitable operators delete for vector deleting
// destructor, we still emit it without errors.

void operator delete(void*, size_t) {}
void operator delete[](void*, size_t) {}

template <typename T> struct RefCounted {
  void operator delete[](void *p) { }
};

struct __declspec(dllexport) DrawingBuffer : public RefCounted<DrawingBuffer> {
  DrawingBuffer();
  virtual ~DrawingBuffer();
};

DrawingBuffer::DrawingBuffer() {}
DrawingBuffer::~DrawingBuffer() {}

struct NoExport : public RefCounted<NoExport> {
  NoExport();
  virtual ~NoExport();
};

NoExport::NoExport() {}
NoExport::~NoExport() {}

namespace std {
  template <class T> struct type_identity {
    typedef T type;
  };
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}
using size_t = __SIZE_TYPE__;

struct Test {
  void operator delete(void *) ;
  virtual ~Test();
};

void *operator new(std::type_identity<Test>, size_t, std::align_val_t) throw();
void operator delete(std::type_identity<Test>, void*, size_t, std::align_val_t) = delete;
void *operator new[](std::type_identity<Test>, size_t, std::align_val_t) throw();
void operator delete[](std::type_identity<Test>, void*, size_t, std::align_val_t) = delete;

void TesttheTest() {

  Test *a = new Test[30];
}

// X64: define dso_local void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// X64: define dso_local void @"??_V@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// X64: define dso_local dllexport void @"??1DrawingBuffer@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %this)

// X64: define weak dso_local noundef ptr @"??_EDrawingBuffer@@UEAAPEAXI@Z"
// X64: call void @"??1DrawingBuffer@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element)
// X64: call void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPEAX@Z"(ptr noundef %2)
// X64: call void @"??_V@YAXPEAX_K@Z"(ptr noundef %2, i64 noundef %{{.*}})
// X64: call void @"??1DrawingBuffer@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// X64: call void @"??3@YAXPEAX_K@Z"(ptr noundef %this1, i64 noundef {{.*}})


// X86: define dso_local void @"??3@YAXPAXI@Z"(ptr noundef %0, i32 noundef %1)
// X86: define dso_local void @"??_V@YAXPAXI@Z"(ptr noundef %0, i32 noundef %1)

// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_EDrawingBuffer@@UAEPAXI@Z"
// X86: call x86_thiscallcc void @"??1DrawingBuffer@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %arraydestroy.element)
// X86: call void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPAX@Z"(ptr noundef %2)
// X86: call void @"??_V@YAXPAXI@Z"(ptr noundef %2, i32 noundef {{.*}})
// X86  call x86_thiscallcc void @"??1DrawingBuffer@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %this1)
// X86: call void @"??3@YAXPAXI@Z"(ptr noundef %this1, i32 noundef {{.*}})


// X64: define weak dso_local noundef ptr @"??_ETest@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(8) %this, i32 noundef %should_call_delete)
// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_ETest@@UAEPAXI@Z"(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %should_call_delete)
// CHECK: dtor.call_delete_after_array_destroy:
// CHECK-NEXT:  call void @llvm.trap()
// CHECK-NEXT:  unreachable
// CHECK: dtor.scalar:
// X64-NEXT: call void @"??1Test@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// X86-NEXT: call x86_thiscallcc void @"??1Test@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %this1)
// CHECK-NEXT: %6 = and i32 %should_call_delete2, 1
// CHECK-NEXT: %7 = icmp eq i32 %6, 0
// CHECK-NEXT: br i1 %7, label %dtor.continue, label %dtor.call_delete
// CHECK: dtor.call_delete:
// X64-NEXT: call void @"??3Test@@SAXPEAX@Z"(ptr noundef %this1)
// X86-NEXT: call void @"??3Test@@SAXPAX@Z"(ptr noundef %this1)
// CHECK-NEXT: br label %dtor.continue

// X64: define linkonce_odr dso_local void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPEAX@Z"(ptr noundef %p)
// X86: define linkonce_odr dso_local void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPAX@Z"(ptr noundef %p)

// X86: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??_GNoExport@@UAEPAXI@Z"(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %should_call_delete)
// X64: define linkonce_odr dso_local noundef ptr @"??_GNoExport@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(8) %this, i32 noundef %should_call_delete)
// CHECK-NOT: define {{.*}}_V{{.*}}NoExport
