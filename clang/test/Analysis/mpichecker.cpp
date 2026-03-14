// RUN: %clang_analyze_cc1 -analyzer-checker=optin.mpi.MPI-Checker -verify %s

#include "MPIMock.h"

void matchedWait1() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank >= 0) {
    MPI_Request sendReq1, recvReq1;
    MPI_Isend(&buf, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &sendReq1);
    MPI_Irecv(&buf, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recvReq1);

    MPI_Wait(&sendReq1, MPI_STATUS_IGNORE);
    MPI_Wait(&recvReq1, MPI_STATUS_IGNORE);
  }
} // no error

void matchedWait2() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank >= 0) {
    MPI_Request sendReq1, recvReq1;
    MPI_Isend(&buf, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &sendReq1);
    MPI_Irecv(&buf, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recvReq1);
    MPI_Wait(&sendReq1, MPI_STATUS_IGNORE);
    MPI_Wait(&recvReq1, MPI_STATUS_IGNORE);
  }
} // no error

void matchedWait3() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank >= 0) {
    MPI_Request sendReq1, recvReq1;
    MPI_Isend(&buf, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &sendReq1);
    MPI_Irecv(&buf, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recvReq1);

    if (rank > 1000) {
      MPI_Wait(&sendReq1, MPI_STATUS_IGNORE);
      MPI_Wait(&recvReq1, MPI_STATUS_IGNORE);
    } else {
      MPI_Wait(&sendReq1, MPI_STATUS_IGNORE);
      MPI_Wait(&recvReq1, MPI_STATUS_IGNORE);
    }
  }
} // no error

void missingWait1() { // Check missing wait for dead region.
  double buf = 0;
  MPI_Request sendReq1;
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &sendReq1);
} // expected-warning{{Request 'sendReq1' has no matching wait.}}

void missingWait2() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
  } else {
    MPI_Request sendReq1, recvReq1;

    MPI_Isend(&buf, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &sendReq1);
    MPI_Irecv(&buf, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recvReq1); // expected-warning{{Request 'sendReq1' has no matching wait.}}
    MPI_Wait(&recvReq1, MPI_STATUS_IGNORE);
  }
}

void doubleNonblocking() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 1) {
  } else {
    MPI_Request sendReq1;

    MPI_Isend(&buf, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &sendReq1);
    MPI_Irecv(&buf, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &sendReq1); // expected-warning{{Double nonblocking on request 'sendReq1'.}}
    MPI_Wait(&sendReq1, MPI_STATUS_IGNORE);
  }
}

void doubleNonblocking2() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Request req;
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &req);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &req); // expected-warning{{Double nonblocking on request 'req'.}}
  MPI_Wait(&req, MPI_STATUS_IGNORE);
}

void doubleNonblocking3() {
  typedef struct { MPI_Request req; } ReqStruct;

  ReqStruct rs;
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &rs.req);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &rs.req); // expected-warning{{Double nonblocking on request 'rs.req'.}}
  MPI_Wait(&rs.req, MPI_STATUS_IGNORE);
}

void doubleNonblocking4() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Request req;
  for (int i = 0; i < 2; ++i) {
    MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &req); // expected-warning{{Double nonblocking on request 'req'.}}
  }
  MPI_Wait(&req, MPI_STATUS_IGNORE);
}

void tripleNonblocking() {
  double buf = 0;
  MPI_Request sendReq;
  MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq);
  MPI_Irecv(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // expected-warning{{Double nonblocking on request 'sendReq'.}}
  MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // expected-warning{{Double nonblocking on request 'sendReq'.}}
  MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
}

void missingNonBlocking() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request sendReq1[10][10][10];
  MPI_Wait(&sendReq1[1][7][9], MPI_STATUS_IGNORE); // expected-warning{{Request 'sendReq1[1][7][9]' has no matching nonblocking call.}}
}

void missingNonBlocking2() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  typedef struct { MPI_Request req[2][2]; } ReqStruct;
  ReqStruct rs;
  MPI_Request *r = &rs.req[0][1];
  MPI_Wait(r, MPI_STATUS_IGNORE); // expected-warning{{Request 'rs.req[0][1]' has no matching nonblocking call.}}
}

void missingNonBlocking3() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request sendReq;
  MPI_Wait(&sendReq, MPI_STATUS_IGNORE); // expected-warning{{Request 'sendReq' has no matching nonblocking call.}}
}

void missingNonBlockingMultiple() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request sendReq[4];
  for (int i = 0; i < 4; ++i) {
    MPI_Wait(&sendReq[i], MPI_STATUS_IGNORE); // expected-warning-re 1+{{Request {{.*}} has no matching nonblocking call.}}
  }
}

void missingNonBlockingWaitall() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request req[4];

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
      &req[0]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
      &req[1]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
      &req[3]);

  MPI_Waitall(4, req, MPI_STATUSES_IGNORE); // expected-warning{{Request 'req[2]' has no matching nonblocking call.}}
}

void missingNonBlockingWaitall2() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request req[4];

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
      &req[0]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
      &req[3]);

  MPI_Waitall(4, req, MPI_STATUSES_IGNORE); // expected-warning-re 2{{Request '{{(.*)[[1-2]](.*)}}' has no matching nonblocking call.}}
}

void missingNonBlockingWaitall3() {
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request req[4];

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
      &req[0]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
      &req[2]);

  MPI_Waitall(4, req, MPI_STATUSES_IGNORE); // expected-warning-re 2{{Request '{{(.*)[[1,3]](.*)}}' has no matching nonblocking call.}}
}

void missingNonBlockingWaitall4() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request req[4];
  MPI_Waitall(4, req, MPI_STATUSES_IGNORE); // expected-warning-re 4{{Request '{{(.*)[[0-3]](.*)}}' has no matching nonblocking call.}}
}

void noDoubleRequestUsage() {
  typedef struct {
    MPI_Request req;
    MPI_Request req2;
  } ReqStruct;

  ReqStruct rs;
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req2);
  MPI_Wait(&rs.req, MPI_STATUS_IGNORE);
  MPI_Wait(&rs.req2, MPI_STATUS_IGNORE);
} // no error

void noDoubleRequestUsage2() {
  typedef struct {
    MPI_Request req[2];
    MPI_Request req2;
  } ReqStruct;

  ReqStruct rs;
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req[0]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req[1]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req2);
  MPI_Wait(&rs.req[0], MPI_STATUS_IGNORE);
  MPI_Wait(&rs.req[1], MPI_STATUS_IGNORE);
  MPI_Wait(&rs.req2, MPI_STATUS_IGNORE);
} // no error

void nestedRequest() {
  typedef struct {
    MPI_Request req[2];
    MPI_Request req2;
  } ReqStruct;

  ReqStruct rs;
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req[0]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req[1]);
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &rs.req2);
  MPI_Waitall(2, rs.req, MPI_STATUSES_IGNORE);
  MPI_Wait(&rs.req2, MPI_STATUS_IGNORE);
} // no error

void singleRequestInWaitall() {
  MPI_Request r;
  int rank = 0;
  double buf = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &r);
  MPI_Waitall(1, &r, MPI_STATUSES_IGNORE);
} // no error

void multiRequestUsage() {
  double buf = 0;
  MPI_Request req;

  MPI_Isend(&buf, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  MPI_Irecv(&buf, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);
} // no error

void multiRequestUsage2() {
  double buf = 0;
  MPI_Request req;

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);
} // no error

// wrapper function
void callNonblocking(MPI_Request *req) {
  double buf = 0;
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
             req);
}

// wrapper function
void callWait(MPI_Request *req) {
  MPI_Wait(req, MPI_STATUS_IGNORE);
}

// Call nonblocking, wait wrapper functions.
void callWrapperFunctions() {
  MPI_Request req;
  callNonblocking(&req);
  callWait(&req);
} // no error

void externFunctions1() {
  double buf = 0;
  MPI_Request req;
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD,
              &req);
  void callWaitExtern(MPI_Request *req);
  callWaitExtern(&req);
} // expected-warning{{Request 'req' has no matching wait.}}

void externFunctions2() {
  MPI_Request req;
  void callNonblockingExtern(MPI_Request *req);
  callNonblockingExtern(&req);
}
