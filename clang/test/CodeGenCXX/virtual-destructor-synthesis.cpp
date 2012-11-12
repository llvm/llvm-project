// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct box {
  virtual ~box();
};

struct pile_box : public box {
  pile_box(box *);
};

pile_box::pile_box(box *pp)
{
}

// CHECK: call void @_ZdlPv

