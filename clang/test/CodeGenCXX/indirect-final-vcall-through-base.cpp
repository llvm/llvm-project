// Actual triple does not matter, just ensuring that the ABI being used for
// mangling and similar is consistent. Choosing x86_64 as that seems to be a
// configured target for most build configurations
// RUN: %clang_cc1 -triple=x86_64 -std=c++26 %s -emit-llvm -O3                          -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64 -std=c++26 %s -emit-llvm -O3 -fstrict-vtable-pointers -o - | FileCheck %s --check-prefix=STRICT

using size_t = unsigned long;
using int64_t = long;

struct Base {
    virtual int64_t sharedGet(size_t i) const = 0;
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
    virtual int64_t useSharedGet(size_t i) {
      return sharedGet(i);
    }
};
  
struct Derived1 final : public Base {
public:
    virtual int64_t sharedGet(size_t i) const override { return 17; }
    int64_t get(size_t i) const override {
        return i;
    }
    
    int64_t getBatch(size_t offset, size_t len, int64_t arr[]) const override;
    virtual int64_t getSumLen(size_t offset, size_t len) const override;
    int64_t directCall(size_t offset, size_t len);
    int64_t directBaseCall(size_t offset, size_t len);
    virtual int64_t useSharedGet(size_t i) override;
};

struct Base2 {
    unsigned value = 0;
    virtual int64_t sharedGet(size_t i) const = 0;
    virtual int64_t get2(size_t i) const = 0;
    virtual int64_t getBatch2(size_t offset, size_t len, int64_t arr[]) const {
      int64_t result = 0;
      for (size_t i = 0; i < len; ++i) {
        result += get2(offset + i);
        arr[i] = get2(offset + i);
      }
      return result;
    }
    virtual int64_t getValue() = 0;
    virtual int64_t callGetValue() {
      return getValue();
    }
    virtual int64_t useBase(Base *b) {
      return b->get(0);
    }
};

struct Derived2 final : Base, Base2 {
    virtual int64_t sharedGet(size_t i) const override { return 19; };
    virtual int64_t get(size_t i) const override {
      return 7;
    };
    virtual int64_t get2(size_t i) const override {
      return 13;
    };
    int64_t getBatch(size_t offset, size_t len, int64_t arr[]) const override;
    virtual int64_t getSumLen(size_t offset, size_t len) const override;
    int64_t getBatch2(size_t offset, size_t len, int64_t arr[]) const override;
    virtual int64_t useSharedGet(size_t i) override;
    virtual int64_t useBase(Base *b) override;
    virtual int64_t getValue() override { return value; }
    virtual int64_t callGetValue() override;
};

struct IntermediateA: virtual Base {

};
struct IntermediateB: virtual Base2 {

};

struct Derived3Part1: IntermediateA {

};

struct Derived3Part2: IntermediateB {

};

struct Derived3 final: Derived3Part1, Derived3Part2 {
    virtual int64_t sharedGet(size_t i) const override { return 23; }
    virtual int64_t get(size_t i) const override { return 27; }
    virtual int64_t getBatch(size_t offset, size_t len, int64_t arr[]) const override;
    virtual int64_t get2(size_t i) const override { return 29; }
    virtual int64_t getBatch2(size_t offset, size_t len, int64_t arr[]) const override;
    virtual int64_t useSharedGet(size_t i) override;
    virtual int64_t useBase(Base *b) override;
    virtual int64_t getValue() override { return value; }
    virtual int64_t callGetValue() override;
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

int64_t Derived1::useSharedGet(size_t i) {
  return Base::useSharedGet(i);
}

int64_t Derived2::getBatch(size_t offset, size_t len, int64_t arr[]) const {
    return Base::getBatch(offset, len, arr);
}

int64_t Derived2::getBatch2(size_t offset, size_t len, int64_t arr[]) const {
    return Base2::getBatch2(offset, len, arr);
}

int64_t Derived2::getSumLen(size_t offset, size_t len) const {
  return Base::getSumLen(offset, len);
}

int64_t Derived2::useSharedGet(size_t i) {
  return Base::useSharedGet(i);
}

int64_t Derived2::useBase(Base *b) {
  return Base2::useBase(this);
}

int64_t Derived2::callGetValue() {
  return Base2::callGetValue();
}

int64_t Derived3::getBatch(size_t offset, size_t len, int64_t arr[]) const {
  return Base::getBatch(offset, len, arr);
}
int64_t Derived3::getBatch2(size_t offset, size_t len, int64_t arr[]) const {
  return Base2::getBatch2(offset, len, arr);
}

int64_t Derived3::useSharedGet(size_t i) {
  return Base::useSharedGet(i);
}
int64_t Derived3::useBase(Base *b) {
  return Base2::useBase(this);
}

int64_t Derived3::callGetValue() {
  return Base2::callGetValue();
}

// CHECK-LABEL: i64 @_ZN8Derived110directCallEmm(
// CHECK: for.body
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE]]
// CHECK: [[VFN:%.*]] = load ptr, ptr [[VFN_SLOT]]
// CHECK: tail call noundef i64 [[VFN]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived19getSumLenEmm(
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE]]
// CHECK: [[VFN:%.*]] = load ptr, ptr [[VFN_SLOT]]
// CHECK: tail call noundef i64 [[VFN]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZN8Derived114directBaseCallEmm(
// CHECK: for.body
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE]]
// CHECK: [[VFN:%.*]] = load ptr, ptr [[VFN_SLOT]]
// CHECK: tail call noundef i64 [[VFN]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived18getBatchEmmPl(
// CHECK: for.
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: tail call noundef i64 [[VFN1]](
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived28getBatchEmmPl(
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: tail call noundef i64 [[VFN1]](
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived29getBatch2EmmPl(
// CHECK: [[OFFSETBASE:%.*]] = getelementptr inbounds nuw i8, ptr %this
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr [[OFFSETBASE]]
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: tail call noundef i64 [[VFN1]](
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr [[OFFSETBASE]]
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZThn8_NK8Derived29getBatch2EmmPl(
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: tail call noundef i64 [[VFN1]](
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived29getSumLenEmm(
// CHECK: for.body
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE]]
// CHECK: [[VFN:%.*]] = load ptr, ptr [[VFN_SLOT]]
// CHECK: tail call noundef i64 [[VFN]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZN8Derived212useSharedGetEm(
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[VFN:%.*]] = load ptr, ptr [[VTABLE]]
// CHECK: tail call noundef i64 [[VFN]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived38getBatchEmmPl
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: tail call noundef i64 [[VFN1]](
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr %this
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZTv0_n40_NK8Derived38getBatchEmmPl
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[THISOFFSET_VSLOT:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 -40
// CHECK: [[THIS_OFFSET:%.*]] = load i64, ptr [[THISOFFSET_VSLOT]]
// CHECK: [[THIS:%.*]] = getelementptr inbounds i8, ptr %this, i64 [[THIS_OFFSET]]
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr [[THIS]]
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr [[THIS]]
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZNK8Derived39getBatch2EmmPl
// CHECK: [[OFFSETBASE:%.*]] = getelementptr inbounds nuw i8, ptr %this
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr [[OFFSETBASE]]
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: tail call noundef i64 [[VFN1]](
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr [[OFFSETBASE]]
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZTv0_n40_NK8Derived39getBatch2EmmPl
// CHECK: entry:
  // %vtable = load ptr, ptr %this, align 8, !tbaa !6
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
  // %0 = getelementptr inbounds i8, ptr %vtable, i64 -40
// CHECK: [[THISOFFSET_VSLOT:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 -40
  // %1 = load i64, ptr %0, align 8
// CHECK: [[THIS_OFFSET:%.*]] = load i64, ptr [[THISOFFSET_VSLOT]]
  // %2 = getelementptr inbounds i8, ptr %this, i64 %1
// CHECK: [[BASE:%.*]] = getelementptr inbounds i8, ptr %this, i64 [[THIS_OFFSET]]
         // %add.ptr.i = getelementptr inbounds nuw i8, ptr %2, i64 16
// CHECK: [[THIS:%.*]] = getelementptr inbounds nuw i8, ptr [[BASE]], i64 16
// CHECK: {{for.body.*:}}
// CHECK: [[VTABLE1:%.*]] = load ptr, ptr [[THIS]]
// CHECK: [[VFN_SLOT1:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE1]]
// CHECK: [[VFN1:%.*]] = load ptr, ptr [[VFN_SLOT1]]
// CHECK: [[VTABLE2:%.*]] = load ptr, ptr [[THIS]]
// CHECK: [[VFN_SLOT2:%.*]] = getelementptr inbounds nuw i8, ptr [[VTABLE2]]
// CHECK: [[VFN2:%.*]] = load ptr, ptr [[VFN_SLOT2]]
// CHECK: tail call noundef i64 [[VFN2]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZN8Derived312useSharedGetEm
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[VFN:%.*]] = load ptr, ptr [[VTABLE]]
// CHECK: tail call noundef i64 [[VFN]](
// CHECK: ret i64

// CHECK-LABEL: i64 @_ZTv0_n56_N8Derived312useSharedGetEm
// CHECK: [[VTABLE:%.*]] = load ptr, ptr %this
// CHECK: [[THISOFFSET_VSLOT:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 -56
// CHECK: [[THIS_OFFSET:%.*]] = load i64, ptr [[THISOFFSET_VSLOT]]
// CHECK: [[THIS:%.*]] = getelementptr inbounds i8, ptr %this, i64 [[THIS_OFFSET]]
// CHECK: [[VTABLE:%.*]] = load ptr, ptr [[THIS]]
// CHECK: [[VFN:%.*]] = load ptr, ptr [[VTABLE]]
// CHECK: tail call noundef i64 [[VFN]](
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

// STRICT-LABEL: i64 @_ZNK8Derived29getSumLenEmm(
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZN8Derived212useSharedGetEm(
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZN8Derived27useBaseEP4Base
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZN8Derived212callGetValueEv(
// STRICT: [[OFFSET_THIS:%.*]] = getelementptr inbounds nuw i8, ptr %this, i64 8
// STRICT: [[INVARIANT_THIS:%.*]] = tail call ptr @llvm.strip.invariant.group.p0(ptr nonnull [[OFFSET_THIS]])
// STRICT: [[VALUE_PTR:%.*]] = getelementptr inbounds nuw i8, ptr [[INVARIANT_THIS]], i64 8
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZThn8_N8Derived212callGetValueEv
// STRICT: [[INVARIANT_THIS:%.*]] = tail call ptr @llvm.strip.invariant.group.p0(ptr nonnull readonly %this)
// STRICT: [[VALUE_PTR:%.*]] = getelementptr inbounds nuw i8, ptr [[INVARIANT_THIS]], i64 8
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZNK8Derived38getBatchEmmPl
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZTv0_n40_NK8Derived38getBatchEmmPl
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZNK8Derived39getBatch2EmmPl
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZTv0_n40_NK8Derived39getBatch2EmmPl
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZN8Derived312useSharedGetEm
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZTv0_n56_N8Derived312useSharedGetEm
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZN8Derived37useBaseEP4Base
// STRICT-NOT: call
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZN8Derived312callGetValueEv(
// STRICT: [[TRUE_THIS:%.*]] = getelementptr inbounds nuw i8, ptr %this, i64 16
// STRICT: [[INVARIANT_THIS:%.*]] = tail call ptr @llvm.strip.invariant.group.p0(ptr nonnull [[TRUE_THIS]])
// STRICT: [[VALUE_PTR:%.*]] = getelementptr inbounds nuw i8, ptr [[INVARIANT_THIS]], i64 8
// STRICT: ret i64

// STRICT-LABEL: i64 @_ZTv0_n56_N8Derived312callGetValueEv
// STRICT: [[VTABLE:%.*]] = load ptr, ptr %this
// STRICT: [[THISOFFSET_VSLOT:%.*]] = getelementptr inbounds i8, ptr [[VTABLE]], i64 -48
// STRICT: [[THIS_OFFSET:%.*]] = load i64, ptr [[THISOFFSET_VSLOT]]
// STRICT: [[VIRTUAL_BASE:%.*]] = getelementptr inbounds i8, ptr %this, i64 [[THIS_OFFSET]]
// STRICT: [[TRUE_THIS:%.*]] = getelementptr inbounds nuw i8, ptr [[VIRTUAL_BASE]], i64 16
// STRICT: [[INVARIANT_THIS:%.*]] = tail call ptr @llvm.strip.invariant.group.p0(ptr nonnull [[TRUE_THIS]])
// STRICT: [[VALUE_PTR:%.*]] = getelementptr inbounds nuw i8, ptr [[INVARIANT_THIS]], i64 8
// STRICT: ret i64
