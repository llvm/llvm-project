template<typename T, typename>
class ClassTmpl { };

typedef int Integer;

template class ClassTmpl<Integer, Integer>;

void f(ClassTmpl<float, Integer> p);

template<typename T>
void g(ClassTmpl<T, T>);

template<> void g<int>(ClassTmpl<int, int>);

// RUN: c-index-test -test-load-source all-display %s | FileCheck %s
// CHECK: print-display-names.cpp:2:7: ClassTemplate=ClassTmpl<T, typename>:2:7
// CHECK: print-display-names.cpp:6:16: ClassDecl=ClassTmpl<Integer, Integer>:6:16 (Definition)
// CHECK: print-display-names.cpp:8:6: FunctionDecl=f(ClassTmpl<float, Integer>):8:6
// CHECK: print-display-names.cpp:11:6: FunctionTemplate=g(ClassTmpl<T, T>):11:6
// CHECK: print-display-names.cpp:13:17: FunctionDecl=g<>(ClassTmpl<int, int>):13:17 [Specialization of g:11:6]
