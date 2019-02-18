// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UninitializedObject \
// RUN:   -analyzer-config alpha.cplusplus.UninitializedObject:Pedantic=true -DPEDANTIC \
// RUN:   -analyzer-config alpha.cplusplus.UninitializedObject:IgnoreRecordsWithField="[Tt]ag|[Kk]ind" \
// RUN:   -std=c++11 -verify  %s

// expected-no-diagnostics

// Both type and name contains "kind".
struct UnionLikeStruct1 {
  enum Kind {
    volume,
    area
  } kind;

  int Volume;
  int Area;

  UnionLikeStruct1(Kind kind, int Val) : kind(kind) {
    switch (kind) {
    case volume:
      Volume = Val;
      break;
    case area:
      Area = Val;
      break;
    }
  }
};

void fUnionLikeStruct1() {
  UnionLikeStruct1 t(UnionLikeStruct1::volume, 10);
}

// Only name contains "kind".
struct UnionLikeStruct2 {
  enum Type {
    volume,
    area
  } kind;

  int Volume;
  int Area;

  UnionLikeStruct2(Type kind, int Val) : kind(kind) {
    switch (kind) {
    case volume:
      Volume = Val;
      break;
    case area:
      Area = Val;
      break;
    }
  }
};

void fUnionLikeStruct2() {
  UnionLikeStruct2 t(UnionLikeStruct2::volume, 10);
}

// Only type contains "kind".
struct UnionLikeStruct3 {
  enum Kind {
    volume,
    area
  } type;

  int Volume;
  int Area;

  UnionLikeStruct3(Kind type, int Val) : type(type) {
    switch (type) {
    case volume:
      Volume = Val;
      break;
    case area:
      Area = Val;
      break;
    }
  }
};

void fUnionLikeStruct3() {
  UnionLikeStruct3 t(UnionLikeStruct3::volume, 10);
}

// Only type contains "tag".
struct UnionLikeStruct4 {
  enum Tag {
    volume,
    area
  } type;

  int Volume;
  int Area;

  UnionLikeStruct4(Tag type, int Val) : type(type) {
    switch (type) {
    case volume:
      Volume = Val;
      break;
    case area:
      Area = Val;
      break;
    }
  }
};

void fUnionLikeStruct4() {
  UnionLikeStruct4 t(UnionLikeStruct4::volume, 10);
}

// Both name and type name contains but does not equal to tag/kind.
struct UnionLikeStruct5 {
  enum WhateverTagBlahBlah {
    volume,
    area
  } FunnyKindName;

  int Volume;
  int Area;

  UnionLikeStruct5(WhateverTagBlahBlah type, int Val) : FunnyKindName(type) {
    switch (FunnyKindName) {
    case volume:
      Volume = Val;
      break;
    case area:
      Area = Val;
      break;
    }
  }
};

void fUnionLikeStruct5() {
  UnionLikeStruct5 t(UnionLikeStruct5::volume, 10);
}
