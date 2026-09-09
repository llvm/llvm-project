// Header for testing late-parsed bounds-safety attributes serialization

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

// Test where counted_by references a field declared later
struct LateRefPointer {
  int *__counted_by(count) buf;
  int count;
};

// Test with sized_by referencing later field
struct LateRefSized {
  int *__sized_by(size) data;
  int size;
};

// Test with counted_by_or_null referencing later field
struct LateRefCountedByOrNull {
  int *__counted_by_or_null(count) buf;
  int count;
};

// Test with sized_by_or_null referencing later field
struct LateRefSizedByOrNull {
  int *__sized_by_or_null(size) data;
  int size;
};

// Test with nested struct
struct LateRefNested {
  struct Inner {
    int value;
  } *__counted_by(n) items;
  int n;
};

// Test with multiple late-parsed attributes
struct MultipleLateRefs {
  int *__counted_by(count1) buf1;
  int *__sized_by(count2) buf2;
  int *__counted_by_or_null(count3) buf3;
  int *__sized_by_or_null(count4) buf4;
  int count1;
  int count2;
  int count3;
  int count4;
};

// Test with anonymous struct/union
struct LateRefAnon {
  int *__counted_by(count) buf;
  struct {
    int count;
  };
};
