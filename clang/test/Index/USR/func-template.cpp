// RUN: c-index-test core -print-source-symbols -- %s | FileCheck %s

template<typename T>
struct A {
  void f(int);
  // CHECK: {{[0-9]+}}:8 | instance-method/C++ | f | c:@ST>1#T@A@F@f#I# |
  
  template<typename U>
  void f(U);
  // CHECK: {{[0-9]+}}:8 | instance-method/C++ | f | c:@ST>1#T@A@FT@>1#Tf#t1.0#v# |
  
  template<>
  void f<int>(int);
  // CHECK: {{[0-9]+}}:8 | instance-method/C++ | f | c:@ST>1#T@A@F@f<#I>#I# |
};
