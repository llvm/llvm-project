// RUN: %clang_analyze_cc1 -analyzer-checker=optin.mpi.MPI-Checker -analyzer-output=text -verify %s

// MPI-Checker test file to test note diagnostics.

#include "MPIMock.h"

void doubleNonblocking() {
  double buf = 0;
  MPI_Request sendReq;
  MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // expected-note{{Request is previously used by nonblocking call here.}}
  MPI_Irecv(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // expected-warning{{Double nonblocking on request 'sendReq'.}} expected-note{{Double nonblocking on request 'sendReq'.}}
  MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
}

void missingWait() {
  double buf = 0;
  MPI_Request sendReq;
  MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &sendReq); // expected-note{{Request is previously used by nonblocking call here.}}
} // expected-warning{{Request 'sendReq' has no matching wait.}} expected-note{{Request 'sendReq' has no matching wait.}}

// If more than 2 nonblocking calls are using a request in a sequence, they all
// point to the first call as the 'previous' call. This is because the
// BugReporterVisitor only checks for differences in state or existence of an
// entity.
void tripleNonblocking() {
  double buf = 0;
  MPI_Request sendReq;
  MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // expected-note 2{{Request is previously used by nonblocking call here.}}
  MPI_Irecv(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // expected-warning{{Double nonblocking on request 'sendReq'.}} expected-note{{Double nonblocking on request 'sendReq'.}}

  MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // expected-warning{{Double nonblocking on request 'sendReq'.}} expected-note{{Double nonblocking on request 'sendReq'.}}

  MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
}
