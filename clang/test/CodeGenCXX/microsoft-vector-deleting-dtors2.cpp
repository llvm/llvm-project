// RUN: %clang_cc1 -emit-llvm -fms-extensions -fms-compatibility -fno-dllexport-inlines %s -triple=x86_64-pc-windows-msvc -o - | FileCheck --check-prefixes=X64,CHECK %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions -fms-compatibility -fno-dllexport-inlines %s -triple=i386-pc-windows-msvc -o - | FileCheck --check-prefixes=X86,CHECK %s

// Check that vector deleting destructor does not reference undefined symbols.

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

// X64: define dso_local void @"??3@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// X64: define dso_local void @"??_V@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// X64: define dso_local dllexport void @"??1DrawingBuffer@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %this)

// X64: define weak dso_local noundef ptr @"??_EDrawingBuffer@@UEAAPEAXI@Z"
// X64: call void @"??1DrawingBuffer@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element)
// X64: call void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPEAX@Z"(ptr noundef %2) #2
// X64: call void @"??_V@YAXPEAX_K@Z"(ptr noundef %2, i64 noundef 8) #3
// X64: call void @"??1DrawingBuffer@@UEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// X64: call void @"??3@YAXPEAX_K@Z"(ptr noundef %this1, i64 noundef 8) #3

// X64: define linkonce_odr dso_local void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPEAX@Z"(ptr noundef %p)

// X86: define dso_local void @"??3@YAXPAXI@Z"(ptr noundef %0, i32 noundef %1)
// X86: define dso_local void @"??_V@YAXPAXI@Z"(ptr noundef %0, i32 noundef %1)

// X86: define weak dso_local x86_thiscallcc noundef ptr @"??_EDrawingBuffer@@UAEPAXI@Z"
// X86: call x86_thiscallcc void @"??1DrawingBuffer@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %arraydestroy.element)
// X86: call void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPAX@Z"(ptr noundef %2)
// X86: call void @"??_V@YAXPAXI@Z"(ptr noundef %2, i32 noundef 4)
// X86  call x86_thiscallcc void @"??1DrawingBuffer@@UAE@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %this1)
// X86: call void @"??3@YAXPAXI@Z"(ptr noundef %this1, i32 noundef 4)

// X86: define linkonce_odr dso_local void @"??_V?$RefCounted@UDrawingBuffer@@@@SAXPAX@Z"(ptr noundef %p)

// X86: define linkonce_odr dso_local x86_thiscallcc noundef ptr @"??_GNoExport@@UAEPAXI@Z"(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %should_call_delete)
// X64: define linkonce_odr dso_local noundef ptr @"??_GNoExport@@UEAAPEAXI@Z"(ptr noundef nonnull align 8 dereferenceable(8) %this, i32 noundef %should_call_delete)
// CHECK-NOT: define {{.*}}_V{{.*}}NoExport
