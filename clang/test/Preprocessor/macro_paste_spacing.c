// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

#define A  x ## y
blah

A
// CHECK: {{^}}xy{{$}}

#define B(x, y) [v ## w] [ v##w] [v##w ] [w ## x] [ w##x] [w##x ] [x ## y] [ x##y] [x##y ] [y ## z] [ y##z] [y##z ]
B(x,y)
// CHECK: [vw] [ vw] [vw ] [wx] [ wx] [wx ] [xy] [ xy] [xy ] [yz] [ yz] [yz ]
B(x,)
// CHECK: [vw] [ vw] [vw ] [wx] [ wx] [wx ] [x] [ x] [x ] [z] [ z] [z ]
B(,y)
// CHECK: [vw] [ vw] [vw ] [w] [ w] [w ] [y] [ y] [y ] [yz] [ yz] [yz ]
B(,)
// CHECK: [vw] [ vw] [vw ] [w] [ w] [w ] [] [ ] [ ] [z] [ z] [z ]

#define C(x, y, z) [x ## y ## z]
C(,,) C(a,,) C(,b,) C(,,c) C(a,b,) C(a,,c) C(,b,c) C(a,b,c)
// CHECK: [] [a] [b] [c] [ab] [ac] [bc] [abc]
