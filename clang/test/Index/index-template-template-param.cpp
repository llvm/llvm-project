// RUN: c-index-test -index-file %s -x objective-c++ | FileCheck %s

template <typename T> class Template1 {};

template <template <class> class TMPL = Template1> class Template2;

// CHECK: [indexEntityReference]: kind: c++-class-template | name: Template1 |
