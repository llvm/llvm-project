// RUN: %clang_cc1 -std=c++26 %s -emit-llvm -O3                          -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++26 %s -emit-llvm -O3 -fstrict-vtable-pointers -o - | FileCheck %s --check-prefix=STRICT

using size_t = unsigned long;
using int64_t = long;

class Base {
  public:
    virtual int64_t get(size_t i) const = 0;
    virtual int64_t getBatch(size_t offset, size_t len, int64_t arr[]) const {
      int64_t result = 0;
      for (size_t i = 0; i < len; ++i) {
        result += get(offset + i);
        arr[i] = get(offset + i);
      }
      return result;
    }
    virtual int64_t getSumLen(size_t offset, size_t len) const {
      int64_t result = 0;
      for (size_t i = 0; i < len; ++i) {
        result += get(offset + i);
      }
      return result;
    }
};
  
class Derived1 final : public Base {
public:
    int64_t get(size_t i) const override {
        return i;
    }
    
    int64_t getBatch(size_t offset, size_t len, int64_t arr[]) const override;
    virtual int64_t getSumLen(size_t offset, size_t len) const override;
    int64_t directCall(size_t offset, size_t len);
    int64_t directBaseCall(size_t offset, size_t len);
};

int64_t Derived1::directCall(size_t offset, size_t len) {
  return getSumLen(offset, len);
}

int64_t Derived1::directBaseCall(size_t offset, size_t len) {
  return Base::getSumLen(offset, len);
}

int64_t Derived1::getBatch(size_t offset, size_t len, int64_t arr[]) const {
    return Base::getBatch(offset, len, arr);
}

int64_t Derived1::getSumLen(size_t offset, size_t len) const {
  return Base::getSumLen(offset, len);
}

// CHECK-LABEL: i64 @_ZN8Derived110directCallEmm(
// CHECK: for.body
// CHECK: %[[VCALL_SLOT1:.*]] = load ptr, ptr %vtable
// CHECK: tail call noundef i64 %[[VCALL_SLOT1]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived19getSumLenEmm
// CHECK: %[[VCALL_SLOT1:.*]] = load ptr, ptr %vtable
// CHECK: tail call noundef i64 %[[VCALL_SLOT1]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZN8Derived114directBaseCallEmm
// CHECK: for.body
// CHECK: %[[VCALL_SLOT1:.*]] = load ptr, ptr %vtable
// CHECK: tail call noundef i64 %[[VCALL_SLOT1]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived18getBatchEmmPl
// CHECK: for.
// CHECK: %[[VCALL_SLOT1:.*]] = load ptr, ptr %vtable
// CHECK: tail call noundef i64 %[[VCALL_SLOT1]](
// CHECK: %[[VCALL_SLOT2:.*]] = load ptr, ptr %vtable
// CHECK: tail call noundef i64 %[[VCALL_SLOT2]](
// CHECK: ret i64


// STRICT-LABEL: i64 @_ZN8Derived110directCallEmm(
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZNK8Derived19getSumLenEmm(
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZN8Derived114directBaseCallEmm(
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZNK8Derived18getBatchEmmPl
// STRICT-NOT: call
// STRICT: ret i64
