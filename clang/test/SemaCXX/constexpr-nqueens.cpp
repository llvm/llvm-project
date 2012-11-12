// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s

typedef unsigned long uint64_t;

struct Board {
  uint64_t State;
  bool Failed;

  constexpr Board() : State(0), Failed(false) {}
  constexpr Board(const Board &O) : State(O.State), Failed(O.Failed) {}
  constexpr Board(uint64_t State, bool Failed = false) :
    State(State), Failed(Failed) {}
  constexpr Board addQueen(int Row, int Col) {
    return Board(State | ((uint64_t)Row << (Col * 4)));
  }
  constexpr int getQueenRow(int Col) {
    return (State >> (Col * 4)) & 0xf;
  }
  constexpr bool ok(int Row, int Col) {
    return okRecurse(Row, Col, 0);
  }
  constexpr bool okRecurse(int Row, int Col, int CheckCol) {
    return Col == CheckCol ? true :
           getQueenRow(CheckCol) == Row ? false :
           getQueenRow(CheckCol) == Row + (Col - CheckCol) ? false :
           getQueenRow(CheckCol) == Row + (CheckCol - Col) ? false :
           okRecurse(Row, Col, CheckCol + 1);
  }
  constexpr bool at(int Row, int Col) {
    return getQueenRow(Col) == Row;
  }
  constexpr bool check(const char *, int=0, int=0);
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
         tryBoard(buildBoardRecurse(N, Col + 1, B.addQueen(Row, Col)),
                  N, Col, Row+1, B) :
         buildBoardScan(N, Col, Row + 1, B);
}
constexpr Board buildBoardRecurse(int N, int Col, const Board &B) {
  return Col == N ? B : buildBoardScan(N, Col, 0, B);
}
constexpr Board buildBoard(int N) {
  return buildBoardRecurse(N, 0, Board());
}

constexpr Board q8 = buildBoard(8);

constexpr bool Board::check(const char *p, int Row, int Col) {
  return
    *p == '\n' ? check(p+1, Row+1, 0) :
    *p == 'o' ? at(Row, Col) && check(p+1, Row, Col+1) :
    *p == '-' ? !at(Row, Col) && check(p+1, Row, Col+1) :
    *p == 0 ? true :
    false;
}
static_assert(q8.check(
    "o-------\n"
    "------o-\n"
    "----o---\n"
    "-------o\n"
    "-o------\n"
    "---o----\n"
    "-----o--\n"
    "--o-----\n"), "");
