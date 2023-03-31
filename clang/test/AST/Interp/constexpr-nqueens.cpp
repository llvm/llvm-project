// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=ref %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fexperimental-new-constant-interpreter -verify %s

/// This is a version of the nqueens.cpp from SemaCXX/,
/// but we don't initialize the State variable in the
/// Board constructors.
/// This tests that InterpFrame::describe().

typedef unsigned long uint64_t;

struct Board {
  uint64_t State;
  bool Failed;

  constexpr Board() : Failed(false) {}
  constexpr Board(const Board &O) : Failed(O.Failed) {}
  constexpr Board(uint64_t State, bool Failed = false) :
    Failed(Failed) {}
  constexpr Board addQueen(int Row, int Col) const {
    return Board(State | ((uint64_t)Row << (Col * 4))); // ref-note {{read of uninitialized object}}
  }
  constexpr int getQueenRow(int Col) const {
    return (State >> (Col * 4)) & 0xf;
  }
  constexpr bool ok(int Row, int Col) const {
    return okRecurse(Row, Col, 0);
  }
  constexpr bool okRecurse(int Row, int Col, int CheckCol) const {
    return Col == CheckCol ? true :
           getQueenRow(CheckCol) == Row ? false :
           getQueenRow(CheckCol) == Row + (Col - CheckCol) ? false :
           getQueenRow(CheckCol) == Row + (CheckCol - Col) ? false :
           okRecurse(Row, Col, CheckCol + 1);
  }
  constexpr bool at(int Row, int Col) const {
    return getQueenRow(Col) == Row;
  }
  constexpr bool check(const char *, int=0, int=0) const;
};

constexpr Board buildBoardRecurse(int N, int Col, const Board &B);
constexpr Board buildBoardScan(int N, int Col, int Row, const Board &B);
constexpr Board tryBoard(const Board &Try,
                         int N, int Col, int Row, const Board &B) {
  return Try.Failed ? buildBoardScan(N, Col, Row, B) : Try;
}
constexpr Board buildBoardScan(int N, int Col, int Row, const Board &B) {
  return Row == N ? Board(0, true) :
         B.ok(Row, Col) ?
         tryBoard(buildBoardRecurse(N, Col + 1, B.addQueen(Row, Col)), // ref-note {{in call to '&Board()->addQueen(0, 0)}}
                  N, Col, Row+1, B) :
         buildBoardScan(N, Col, Row + 1, B);
}
constexpr Board buildBoardRecurse(int N, int Col, const Board &B) {
  return Col == N ? B : buildBoardScan(N, Col, 0, B); // ref-note {{in call to 'buildBoardScan(8, 0, 0, Board())'}}
}
constexpr Board buildBoard(int N) {
  return buildBoardRecurse(N, 0, Board()); // ref-note {{in call to 'buildBoardRecurse(8, 0, Board())'}}
}

constexpr Board q8 = buildBoard(8); // ref-error {{must be initialized by a constant expression}} \
                                    // ref-note {{in call to 'buildBoard(8)'}} \
                                    // expected-error {{must be initialized by a constant expression}}
