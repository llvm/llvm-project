// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -verify %s

// Vertex return declarations have the same interpolation restrictions as pixel
// inputs. Check the return type, not the function type.
[shader("vertex")]
// expected-error@+1 {{cannot be used with type 'uint'}}
linear uint invalid_integer() : VALUE { return 0; }
[shader("vertex")]
// expected-error@+1 {{cannot be used with type 'bool'}}
centroid bool invalid_bool() : VALUE { return false; }
[shader("vertex")]
// expected-error@+1 {{cannot be used with type 'double'}}
sample double invalid_double() : VALUE { return 0; }
[shader("vertex")]
// expected-error@+2 {{'nointerpolation' cannot be used on SV_Position}}
// expected-note@+1 {{conflicting attribute is here}}
nointerpolation float4 invalid_position() : SV_Position { return 0; }

// Unqualified aggregates must still validate modifiers on nested fields.
struct InvalidField {
  // expected-error@+1 {{cannot be used with type 'uint'}}
  linear uint value : VALUE;
};
struct InvalidOuter {
  float4 pos : SV_Position;
  InvalidField inner;
};
[shader("vertex")]
InvalidOuter invalid_nested_field() { return (InvalidOuter)0; }

// A return modifier propagates through nested aggregates and array leaves.
struct IntegerLeaf {
  int values[2] : VALUE;
};
struct IntegerOuter {
  IntegerLeaf inner;
};
[shader("vertex")]
// expected-error@+1 {{cannot be used with type 'int'}}
linear IntegerOuter invalid_inherited_type() { return (IntegerOuter)0; }

struct PositionLeaf {
  // expected-note@+1 {{conflicting attribute is here}}
  float4 pos : SV_Position;
};
struct PositionOuter {
  PositionLeaf inner;
};
[shader("vertex")]
// expected-error@+1 {{'nointerpolation' cannot be used on SV_Position}}
nointerpolation PositionOuter invalid_inherited_position() {
  return (PositionOuter)0;
}

struct FlatPosition {
  // expected-error@+2 {{'nointerpolation' cannot be used on SV_Position}}
  // expected-note@+1 {{conflicting attribute is here}}
  nointerpolation float4 pos : SV_Position;
};
[shader("vertex")]
FlatPosition invalid_position_field() { return (FlatPosition)0; }

// The output side of out and inout parameters also needs validation.
[shader("vertex")]
void invalid_output_parameters(
    // expected-error@+1 {{cannot be used with type 'uint'}}
    linear out uint a : A,
    // expected-error@+1 {{cannot be used with type 'int'}}
    sample inout int b : B) {
  a = 0;
}

// Vertex inputs and non-entry declarations still ignore these restrictions.
linear uint helper() { return 0; }
[shader("vertex")]
float4 valid_input(linear uint a : VALUE) : SV_Position { return 0; }

[shader("vertex")]
nointerpolation uint valid_integer() : VALUE { return 0; }
[shader("vertex")]
nointerpolation bool valid_bool() : VALUE { return false; }
[shader("vertex")]
nointerpolation double valid_double() : VALUE { return 0; }

// Inner modifiers replace the entire inherited set, including the return's
// modifier. Neither the integer nor the position should inherit sample/flat.
struct OverrideLeaf {
  nointerpolation uint value : VALUE;
  linear float4 pos : SV_Position;
};
struct OverrideOuter {
  nointerpolation OverrideLeaf inner;
  float other : OTHER;
};
[shader("vertex")]
sample OverrideOuter valid_overrides() { return (OverrideOuter)0; }

[shader("vertex")]
void valid_output_parameters(nointerpolation out uint a : A,
                             nointerpolation inout int b : B) {
  a = 0;
}
