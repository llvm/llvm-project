// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// Test that counted_by works correctly with late parsing when ActOnFields
// is called before late-parsed attributes are evaluated. This allows offset
// expressions in counted_by to be evaluated.

#define __counted_by(f)  __attribute__((counted_by(f)))

struct size_known {
  int field;
};

//==============================================================================
// Verify anonymous struct handling works corrctly under currect ordering.
// GetEnclosingNamedOrTopAnonRecord must correctly walk through anonymous
// structs when the struct is already marked complete.
//==============================================================================

// count in outer struct, buf in anonymous struct
struct on_pointer_anon_buf {
  int count;
  struct {
    struct size_known *buf __counted_by(count);
  };
};

// both count and buf in anonymous struct
struct on_pointer_anon_both {
  struct {
    int count;
    struct size_known *buf __counted_by(count);
  };
};

// nested anonymous structs
struct on_pointer_nested_anon {
  int count;
  struct {
    struct {
      struct size_known *buf __counted_by(count);
    };
  };
};

//==============================================================================
// Verify non-anonymous unnamed structs correctly reject counted_by if it
// reference fields in the outer struct.
//==============================================================================

// count in outer, buf in non-anonymous unnamed struct — should reject
struct on_pointer_named_inner {
  int count; // expected-note{{'count' declared here}}
  struct {
		// expected-error@+1{{'counted_by' field 'count' isn't within the same struct as the annotated pointer}}
    struct size_known *buf __counted_by(count); 
  } inner;
};

// both in non-anonymous unnamed struct — should accept
struct on_pointer_named_inner_both {
  struct {
    int count;
    struct size_known *buf __counted_by(count);
  } inner;
};
