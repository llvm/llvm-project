// RUN: %clang_cc1 %s -fopenacc -verify

struct NotConvertible{} NC;
short getS();
int getI();

void uses() {
  int arr[5];

#pragma acc data copyin(arr[0]) wait
  ;

#pragma acc enter data copyin(arr[0]) wait()

#pragma acc exit data copyout(arr[0]) wait(getS(), getI())

  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'host_data' directive}}
#pragma acc host_data use_device(arr) wait(getS(), getI())
  ;

#pragma acc data copyin(arr[0]) wait(devnum:getS(): getI())
  ;

#pragma acc enter data copyin(arr[0]) wait(devnum:getS(): queues: getI()) wait(devnum:getI(): queues: getS(), getI(), 5)

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc exit data copyout(arr[0]) wait(devnum:NC : 5)

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc data copyin(arr[0]) wait(devnum:5 : NC)
  ;

  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('int[5]' invalid)}}
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc enter data copyin(arr[0]) wait(devnum:arr : queues: arr, NC, 5)

  // expected-error@+1{{OpenACC 'wait' clause is not valid on 'loop' directive}}
#pragma acc loop wait
  for(int i = 5; i < 10;++i);
}
