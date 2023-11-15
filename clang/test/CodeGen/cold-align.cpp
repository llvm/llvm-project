// Dont alter function alignment if marked cold
//
// Cold attribute marks functions as also optimize for size. This normally collapses the
// default function alignment. This can interfere with edit&continue effectiveness.
//
// RUN:     %clang -O2 -S -emit-llvm %s -o - | FileCheck %s -check-prefixes TWO
// RUN:     %clang -Os -S -emit-llvm %s -o - | FileCheck %s -check-prefixes SIZE

class Dismissed
{
public:
  __attribute__((cold)) void Chilly();
                        void Temparate();
  __attribute__((hot))  void Sizzle();
};
void Dismissed::Chilly(){};
void Dismissed::Temparate(){};
void Dismissed::Sizzle(){};

// TWO: attributes #0 = {
// TWO: "keepalign"="true"
// TWO: attributes #1 = {
// TWO-NOT: "keepalign"="true"
// TWO: attributes #2 = {
// TWO-NOT: "keepalign"="true"

// SIZE: attributes #0 = {
// SIZE-NOT: "keepalign"="true"
// SIZE: attributes #1 = {
// SIZE-NOT: "keepalign"="true"
// SIZE: attributes #2 = {
// SIZE-NOT: "keepalign"="true"