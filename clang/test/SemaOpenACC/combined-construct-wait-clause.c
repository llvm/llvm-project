// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
short getS();
int getI();

void uses() {
  int arr[5];

#pragma acc parallel loop wait
  for (unsigned i = 0; i < 5; ++i);

#pragma acc serial loop wait()
  for (unsigned i = 0; i < 5; ++i);

#pragma acc kernels loop wait(getS(), getI())
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop wait(devnum:getS(): getI())
  for (unsigned i = 0; i < 5; ++i);

#pragma acc parallel loop wait(devnum:getS(): queues: getI()) wait(devnum:getI(): queues: getS(), getI(), 5)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel loop wait(devnum:NC : 5)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel loop wait(devnum:5 : NC)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel loop wait(devnum:arr : queues: arr, NC, 5)
  for (unsigned i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait
  for(int i = 5; i < 10;++i);
}
