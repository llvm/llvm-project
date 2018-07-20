template<typename T, typename>
class ClassTmpl { };

typedef int Integer;

template class ClassTmpl<Integer, Integer>;

void f(ClassTmpl<float, Integer> p);

template<typename T>
void g(ClassTmpl<T, T>);

template<> void g<int>(ClassTmpl<int, int>);

// RUN: c-index-test -test-load-source all-display %s | FileCheck %s --check-prefix=DISPLAY_NAME
// DISPLAY_NAME: print-display-names.cpp:2:7: ClassTemplate=ClassTmpl<T, typename>:2:7
// DISPLAY_NAME: print-display-names.cpp:6:16: ClassDecl=ClassTmpl<Integer, Integer>:6:16 (Definition)
// DISPLAY_NAME: print-display-names.cpp:8:6: FunctionDecl=f(ClassTmpl<float, Integer>):8:6
// DISPLAY_NAME: print-display-names.cpp:11:6: FunctionTemplate=g(ClassTmpl<T, T>):11:6
// DISPLAY_NAME: print-display-names.cpp:13:17: FunctionDecl=g<>(ClassTmpl<int, int>):13:17 [Specialization of g:11:6]

// RUN: env CINDEXTEST_PRINTINGPOLICY_TERSEOUTPUT=1 c-index-test -test-load-source all-pretty %s | FileCheck %s --check-prefix=PRETTY
// PRETTY: print-display-names.cpp:2:7: ClassTemplate=template <typename T, typename > class ClassTmpl {}:2:7 (Definition) Extent=[1:1 - 2:20]
// PRETTY: print-display-names.cpp:4:13: TypedefDecl=typedef int Integer:4:13 (Definition) Extent=[4:1 - 4:20]
// PRETTY: print-display-names.cpp:6:16: ClassDecl=template<> class ClassTmpl<int, int> {}:6:16 (Definition) [Specialization of ClassTmpl:2:7] Extent=[6:1 - 6:43]
// PRETTY: print-display-names.cpp:8:6: FunctionDecl=void f(ClassTmpl<float, Integer> p):8:6 Extent=[8:1 - 8:36]
// PRETTY: print-display-names.cpp:8:34: ParmDecl=ClassTmpl<float, Integer> p:8:34 (Definition) Extent=[8:8 - 8:35]
// PRETTY: print-display-names.cpp:11:6: FunctionTemplate=template <typename T> void g(ClassTmpl<T, T>):11:6 Extent=[10:1 - 11:24]
// PRETTY: print-display-names.cpp:11:23: ParmDecl=ClassTmpl<T, T>:11:23 (Definition) Extent=[11:8 - 11:23]
// PRETTY: print-display-names.cpp:13:17: FunctionDecl=template<> void g<int>(ClassTmpl<int, int>):13:17 [Specialization of g:11:6] [Template arg 0: kind: 1, type: int] Extent=[13:1 - 13:44]
// PRETTY: print-display-names.cpp:13:43: ParmDecl=ClassTmpl<int, int>:13:43 (Definition) Extent=[13:24 - 13:43]
