// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
short getS();
int getI();

void uses() {
  int arr[5];

#pragma acc parallel wait
  while(1);

#pragma acc serial wait()
  while(1);

#pragma acc kernels wait(getS(), getI())
  while(1);

#pragma acc parallel wait(devnum:getS(): getI())
  while(1);

#pragma acc parallel wait(devnum:getS(): queues: getI()) wait(devnum:getI(): queues: getS(), getI(), 5)
  while(1);

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel wait(devnum:NC : 5)
  while(1);

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel wait(devnum:5 : NC)
  while(1);

  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc parallel wait(devnum:arr : queues: arr, NC, 5)
  while(1);

  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait
  for(;;);
}
