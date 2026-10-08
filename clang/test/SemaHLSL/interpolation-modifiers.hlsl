// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -verify %s

// Each keyword is incompatible with nointerpolation, in either order.
void conflicts(
    // expected-error@+2 {{'nointerpolation' cannot be combined}}
    // expected-note@+1 {{conflicting attribute is here}}
    nointerpolation linear float a,
    // expected-error@+2 {{'nointerpolation' cannot be combined}}
    // expected-note@+1 {{conflicting attribute is here}}
    centroid nointerpolation float b,
    // expected-error@+2 {{'nointerpolation' cannot be combined}}
    // expected-note@+1 {{conflicting attribute is here}}
    nointerpolation noperspective float c,
    // expected-error@+2 {{'nointerpolation' cannot be combined}}
    // expected-note@+1 {{conflicting attribute is here}}
    sample nointerpolation float d,
    // expected-error@+2 {{'nointerpolation' cannot be combined}}
    // expected-note@+1 {{conflicting attribute is here}}
    nointerpolation center float e);

void duplicates(
    // expected-warning@+1 {{duplicate interpolation modifier 'nointerpolation'}}
    nointerpolation nointerpolation float a,
    // expected-warning@+1 {{duplicate interpolation modifier 'linear'}}
    linear linear float b,
    // expected-warning@+1 {{duplicate interpolation modifier 'centroid'}}
    centroid centroid float c,
    // expected-warning@+1 {{duplicate interpolation modifier 'noperspective'}}
    noperspective noperspective float d,
    // expected-warning@+1 {{duplicate interpolation modifier 'sample'}}
    sample sample float e,
    // expected-warning@+1 {{duplicate interpolation modifier 'center'}}
    center center float f);

void locations(
    // expected-warning@+1 {{interpolation modifier 'centroid' overrides 'center'}}
    center centroid float a,
    // expected-warning@+1 {{interpolation modifier 'centroid' overrides 'center'}}
    centroid center float b,
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'center'}}
    center sample float c,
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'center'}}
    sample center float d,
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'centroid'}}
    centroid sample float e,
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'centroid'}}
    sample centroid float f);

// With all three locations, compare the new keyword against the previous
// highest-precedence location, regardless of keyword order.
void all_locations(
    // expected-warning@+2 {{interpolation modifier 'centroid' overrides 'center'}}
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'centroid'}}
    center centroid sample float a,
    // expected-warning@+2 {{interpolation modifier 'sample' overrides 'center'}}
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'centroid'}}
    center sample centroid float b,
    // expected-warning@+2 {{interpolation modifier 'centroid' overrides 'center'}}
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'centroid'}}
    centroid center sample float c,
    // expected-warning@+2 {{interpolation modifier 'sample' overrides 'centroid'}}
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'center'}}
    centroid sample center float d,
    // expected-warning@+2 {{interpolation modifier 'sample' overrides 'center'}}
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'centroid'}}
    sample center centroid float e,
    // expected-warning@+2 {{interpolation modifier 'sample' overrides 'centroid'}}
    // expected-warning@+1 {{interpolation modifier 'sample' overrides 'center'}}
    sample centroid center float f);

// Interpolation modes do not imply an explicit sampling location, so combining
// them with a location must not produce an override warning in either order.
void implicit_locations(linear centroid float a, centroid linear float b,
                        noperspective sample float c, sample noperspective float d,
                        linear center float e, center linear float f);

// Type restrictions apply to pixel shader inputs, not arbitrary declarations.
[shader("pixel")]
float4 invalid_types(
    // expected-error@+1 {{cannot be used with type 'int'}}
    linear int a : A,
    // expected-error@+1 {{cannot be used with type 'uint'}}
    centroid uint2 b : B,
    // expected-error@+1 {{cannot be used with type 'bool'}}
    noperspective bool c : C,
    // expected-error@+1 {{cannot be used with type 'int'}}
    center int e[2][3] : E,
    // expected-error@+1 {{cannot be used with type 'int'}}
    linear int2x2 f : F,
    // expected-error@+1 {{cannot be used with type 'int'}}
    linear in int g : G) : SV_Target {
  return 0;
}

[shader("pixel")]
float4 valid_types(nointerpolation int a : A, nointerpolation bool2 b : B,
                   nointerpolation int d[2] : D,
                   nointerpolation int2x2 e : E, linear float f : F,
                   centroid half2 g : G, sample float2x2 h : H) : SV_Target {
  return 0;
}

// The input side of an inout parameter also needs interpolation validation.
// SV_Target currently rejects integer types independently of interpolation.
[shader("pixel")]
// expected-error@+2 {{cannot be used with type 'uint'}}
// expected-error@+1 {{attribute 'SV_Target' only applies to a field or parameter of type}}
void invalid_inout(linear inout uint a : SV_Target) {
  a = 0;
}

struct Position {
  // expected-note@+1 {{conflicting attribute is here}}
  float4 p : SV_Position;
};
[shader("pixel")]
// expected-error@+1 {{'nointerpolation' cannot be used on SV_Position}}
float4 inherited_position(nointerpolation Position p) : SV_Target {
  return 0;
}
[shader("pixel")]
// expected-error@+2 {{'nointerpolation' cannot be used on SV_Position}}
// expected-note@+1 {{conflicting attribute is here}}
float4 direct_position(nointerpolation float4 p : SV_Position) : SV_Target {
  return 0;
}

struct Inner {
  int value;
};
struct Outer {
  Inner inner;
};
[shader("pixel")]
// expected-error@+1 {{cannot be used with type 'int'}}
float4 invalid_aggregate(linear Outer x : A) : SV_Target {
  return 0;
}

// Field modifiers are checked even without an enclosing modifier, including
// through unannotated aggregates. The field's declaration is not itself invalid.
struct FieldModifiers {
  // expected-error@+1 {{cannot be used with type 'uint'}}
  sample uint value;
};
struct FieldOuter {
  FieldModifiers inner;
};
[shader("pixel")]
float4 invalid_field_modifier(FieldOuter x : A) : SV_Target { return 0; }
[shader("vertex")]
float4 valid_field_modifier(FieldOuter x : A) : SV_Position { return 0; }

struct FlatPosition {
  // expected-error@+2 {{'nointerpolation' cannot be used on SV_Position}}
  // expected-note@+1 {{conflicting attribute is here}}
  nointerpolation float4 p : SV_Position;
};
[shader("pixel")]
float4 invalid_position_field(FlatPosition p) : SV_Target { return 0; }

// An inner modifier replaces the entire inherited set, not just one keyword.
struct Overrides {
  nointerpolation int i;
  linear float f;
};
[shader("pixel")]
float4 valid_aggregate(sample Overrides x : A) : SV_Target { return 0; }
struct PositionOverride {
  linear float4 p : SV_Position;
};
[shader("pixel")]
float4 valid_position(nointerpolation PositionOverride p) : SV_Target { return 0; }

// Subject restrictions also apply outside entry points.
// expected-error@+1 {{'linear' attribute only applies to parameters, non-static data members, and functions}}
linear float global;
// expected-error@+1 {{'sample' attribute only applies to parameters, non-static data members, and functions}}
typedef sample float Alias;
void local() {
  // expected-error@+1 {{'centroid' attribute only applies to parameters, non-static data members, and functions}}
  centroid float value;
}
