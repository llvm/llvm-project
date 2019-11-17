// Verify proper creation of sync regions and syncs when _Cilk_spawn statements
// are nested within a _Cilk_for.
//
// Thanks to Jackie Bredenberg and Yuan Yao for the original source code for
// this test case.
//
// RUN: %clang_cc1 %s -std=gnu11 -triple x86_64-unknown-linux-gnu -O0 -fcilkplus -verify -S -emit-llvm -ftapir=none -o - | FileCheck %s
// expected-no-diagnostics

// A two-dimensional line.
typedef struct Line {
  unsigned char quadrant;  // stores 4 bits about position within the quadtree
} Line;

typedef struct LineList {
  unsigned int size;
} LineList;

typedef struct Quadtree {
  // If this quadtree is not expanded, then all four below are NULL
  struct Quadtree* LD;  // x < center.x, y < center.y
  struct Quadtree* LU;  // x < center.x, y >= center.y
  struct Quadtree* RD;  // x >= center.x, y < center.y
  struct Quadtree* RU;  // x >= center.x, y >= center.y

  // A list of lines that do not fit inside any subtree
  LineList* lines;

  // An array version of LineList, for parallelization.
  Line** lineArray;

  // Number of lines in the quadtree (including ones in the subtrees)
  unsigned int numLines;
} Quadtree;

typedef struct CollisionWorld CollisionWorld;

#define LD_BIT 0b0001
#define LU_BIT 0b0010
#define RD_BIT 0b0100
#define RU_BIT 0b1000

void Quadtree_findAllIntersectingPairsWithLine(Quadtree* quadtree,
                                               CollisionWorld* collisionWorld,
                                               Line* line);

// search through the quadtree to find any collisions contained within it,
// including collisions with lines in the sub-trees
void Quadtree_findAllIntersectingPairs(Quadtree* quadtree,
                                       CollisionWorld* collisionWorld) {
  LineList* currentLevelLines = quadtree->lines;
  int nodeSize = currentLevelLines->size;

  // For each line, check it with the subtrees if there are possible intersections.
  _Cilk_for (int i = 0; i < nodeSize; i++) {
    // CHECK: detach within %[[OUTERSR:.+]], label %{{.+}}, label
    Line* line = quadtree->lineArray[i];

    // Checking if we need to check this line against each of the subtrees
    // If none of the four sides of the parallelogram swept out by the line
    // intersect any of the four sides and is not inside, then don't bother checking.
    if (quadtree->LD->numLines > 0 && (line->quadrant & LD_BIT)) {
      _Cilk_spawn Quadtree_findAllIntersectingPairsWithLine(quadtree->LD, collisionWorld, line);
      // CHECK: detach within %[[INNERSR:.+]], label %{{.+}}, label
    }
    if (quadtree->LU->numLines > 0 && (line->quadrant & LU_BIT)) {
      _Cilk_spawn Quadtree_findAllIntersectingPairsWithLine(quadtree->LU, collisionWorld, line);
      // CHECK: detach within %[[INNERSR]],
    }
    if (quadtree->RD->numLines > 0 && (line->quadrant & RD_BIT)) {
      _Cilk_spawn Quadtree_findAllIntersectingPairsWithLine(quadtree->RD, collisionWorld, line);
      // CHECK: detach within %[[INNERSR]],
    }
    if (quadtree->RU->numLines > 0 && (line->quadrant & RU_BIT)) {
      _Cilk_spawn Quadtree_findAllIntersectingPairsWithLine(quadtree->RU, collisionWorld, line);
    // CHECK: detach within %[[INNERSR]],
    }
    // CHECK: sync within %[[INNERSR]],
    // CHECK: reattach within %[[OUTERSR]],
  }
  // CHECK: sync within %[[OUTERSR]],

  // Find intersections within the subtrees
  if (quadtree->LD->numLines > 1) {
    _Cilk_spawn Quadtree_findAllIntersectingPairs(quadtree->LD, collisionWorld);
    // CHECK: detach within %[[OUTERSR2:.+]], label %{{.+}}, label
  }
  if (quadtree->LU->numLines > 1) {
    _Cilk_spawn Quadtree_findAllIntersectingPairs(quadtree->LU, collisionWorld);
    // CHECK: detach within %[[OUTERSR2]],
  }
  if (quadtree->RD->numLines > 1) {
    _Cilk_spawn Quadtree_findAllIntersectingPairs(quadtree->RD, collisionWorld);
    // CHECK: detach within %[[OUTERSR2]],
  }
  if (quadtree->RU->numLines > 1) {
    Quadtree_findAllIntersectingPairs(quadtree->RU, collisionWorld);
  }
  _Cilk_sync;
  // CHECK: sync within %[[OUTERSR2]],
}
