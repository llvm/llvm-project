// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck %s

// Test that vector deleting destructors are emitted when new[] is used,
// even when the destructor definition is in another translation unit.

struct ForwardDeclared {
  ForwardDeclared();
  virtual ~ForwardDeclared();
};

struct DefinedInTU {
  virtual ~DefinedInTU();
};

struct NonVirtualDtor {
  ~NonVirtualDtor();
};

struct NoDtor {
  virtual void foo();
  int x;
};

struct DeclDerived : ForwardDeclared {
  ~DeclDerived() override;
};

struct InlineDefaulted {
  virtual ~InlineDefaulted() = default;
};

struct OutOfLineDefaulted {
  virtual ~OutOfLineDefaulted();
};

OutOfLineDefaulted::~OutOfLineDefaulted() = default;

template<typename T>
struct Container {
  T data;
  virtual ~Container();
};

extern template class Container<int>;
Container<int> *arr = new Container<int>[5];

struct ImplicitVDtorDerived : ForwardDeclared{
  int data;
};

struct __declspec(dllimport) DllImported {
  virtual ~DllImported();
};

struct VirtualDerived : virtual ForwardDeclared {
  ~VirtualDerived() override;
};

struct DeclaredCtorDefinedDtor {
  DeclaredCtorDefinedDtor();
  virtual ~DeclaredCtorDefinedDtor() {}
};

struct TemplateNotAllocated {
  TemplateNotAllocated();
  virtual ~TemplateNotAllocated();
};

struct TemplateAllocated {
  TemplateAllocated();
  virtual ~TemplateAllocated();
};

template <int T>
void allocate() {
  TemplateNotAllocated *arr = new TemplateNotAllocated[T];
}

template <typename T>
void actuallyAllocate() {
  T *arr = new T[10];
  delete[] arr;
}

void cases() {
  ForwardDeclared *arr = new ForwardDeclared[5];
  DefinedInTU *arr1 = new DefinedInTU[5];
  NonVirtualDtor *arr2 = new NonVirtualDtor[5];
  NoDtor *arr3 = new NoDtor[5];
  ForwardDeclared *arr4 = new DeclDerived[5];
  InlineDefaulted *arr5 = new InlineDefaulted[5];
  OutOfLineDefaulted *arr6 = new OutOfLineDefaulted[5];
  ImplicitVDtorDerived *arr7 = new ImplicitVDtorDerived[5];
  DllImported *arr8 = new DllImported[5];
  VirtualDerived *arr9 = new VirtualDerived[3];
  DeclaredCtorDefinedDtor *arr10 = new DeclaredCtorDefinedDtor[5];
  actuallyAllocate<TemplateAllocated>();
}


// CHECK-DAG: declare dso_local void @"??1ForwardDeclared@@UEAA@XZ"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EForwardDeclared@@UEAAPEAXI@Z"(
// CHECK-DAG: define dso_local void @"??1DefinedInTU@@UEAA@XZ"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EDefinedInTU@@UEAAPEAXI@Z"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EDeclDerived@@UEAAPEAXI@Z"(
// CHECK-DAG: declare dso_local void @"??1DeclDerived@@UEAA@XZ"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EInlineDefaulted@@UEAAPEAXI@Z"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EOutOfLineDefaulted@@UEAAPEAXI@Z"(
// CHECK-DAG: declare dso_local void @"??1?$Container@H@@UEAA@XZ"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_E?$Container@H@@UEAAPEAXI@Z"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EImplicitVDtorDerived@@UEAAPEAXI@Z"(
// CHECK-DAG: declare dllimport void @"??1DllImported@@UEAA@XZ"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EDllImported@@UEAAPEAXI@Z"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EVirtualDerived@@UEAAPEAXI@Z"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_EDeclaredCtorDefinedDtor@@UEAAPEAXI@Z"(
// CHECK-DAG: declare dso_local void @"??1TemplateAllocated@@UEAA@XZ"(
// CHECK-DAG: define weak dso_local noundef ptr @"??_ETemplateAllocated@@UEAAPEAXI@Z"(
// CHECK-NOT: @"??_ETemplateNotAllocated@@
// CHECK-NOT: @"??_ENonVirtualDtor@@
// CHECK-NOT: @"??_ENoDtor@@

DefinedInTU::~DefinedInTU() {}
