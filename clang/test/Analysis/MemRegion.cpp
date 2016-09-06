// RUN: %clang_cc1 -analyze -analyzer-checker=optin.mpi.MPI-Checker -verify %s

#include "MPIMock.h"

// Use MPI-Checker to test 'getDescriptiveName', as the checker uses the
// function for diagnostics.
void testGetDescriptiveName() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request sendReq1;
  MPI_Wait(&sendReq1, MPI_STATUS_IGNORE); // expected-warning{{Request 'sendReq1' has no matching nonblocking call.}}
}

void testGetDescriptiveName2() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Request sendReq1[10][10][10];
  MPI_Wait(&sendReq1[1][7][9], MPI_STATUS_IGNORE); // expected-warning{{Request 'sendReq1[1][7][9]' has no matching nonblocking call.}}
}

void testGetDescriptiveName3() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  typedef struct { MPI_Request req; } ReqStruct;
  ReqStruct rs;
  MPI_Request *r = &rs.req;
  MPI_Wait(r, MPI_STATUS_IGNORE); // expected-warning{{Request 'rs.req' has no matching nonblocking call.}}
}

void testGetDescriptiveName4() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  typedef struct { MPI_Request req[2][2]; } ReqStruct;
  ReqStruct rs;
  MPI_Request *r = &rs.req[0][1];
  MPI_Wait(r, MPI_STATUS_IGNORE); // expected-warning{{Request 'rs.req[0][1]' has no matching nonblocking call.}}
}

void testGetDescriptiveName5() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  typedef struct { MPI_Request req; } ReqStructInner;
  typedef struct  { ReqStructInner req; } ReqStruct;
  ReqStruct rs;
  MPI_Request *r = &rs.req.req;
  MPI_Wait(r, MPI_STATUS_IGNORE); // expected-warning{{Request 'rs.req.req' has no matching nonblocking call.}}
}
