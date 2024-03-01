// RUN: %clang_cc1 %s -verify -fopenacc

void BreakContinue() {

#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    switch(i) {
      case 0:
      break; // leaves switch, not 'for'.
      default:
      i +=2;
      break;
    }
    if (i == 2)
      continue;

    break;  // expected-error{{invalid branch out of OpenACC Compute Construct}}
  }

  int j;
  switch(j) {
    case 0:
#pragma acc parallel
    {
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
    }
    case 1:
#pragma acc parallel
    {
    }
    break;
  }

#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
    if (i > 1)
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
  }

#pragma acc parallel
  switch(j) {
    case 1:
      break;
  }

#pragma acc parallel
  {
    for(int i = 1; i < 100; i++) {
      if (i > 4)
        break;
    }
  }

  for (int i =0; i < 5; ++i) {
#pragma acc parallel
    {
      continue; // expected-error{{invalid branch out of OpenACC Compute Construct}}
    }
  }

#pragma acc parallel
  for (int i =0; i < 5; ++i) {
    continue;
  }

#pragma acc parallel
  for (int i =0; i < 5; ++i) {
    {
      continue;
    }
  }

  for (int i =0; i < 5; ++i) {
#pragma acc parallel
    {
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
    }
  }

#pragma acc parallel
  while (j) {
    --j;
    if (j > 4)
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
  }

#pragma acc parallel
  do {
    --j;
    if (j > 4)
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}
  } while (j );

}

void Return() {
#pragma acc parallel
  {
    return;// expected-error{{invalid return out of OpenACC Compute Construct}}
  }

#pragma acc parallel
  {
    {
      return;// expected-error{{invalid return out of OpenACC Compute Construct}}
    }
  }

#pragma acc parallel
  {
    for (int i = 0; i < 5; ++i) {
      return;// expected-error{{invalid return out of OpenACC Compute Construct}}
    }
  }
}

void Goto() {
  int j;
#pragma acc parallel // expected-note{{invalid branch out of OpenACC Compute Construct}}
  while(j) {
    if (j <3)
      goto LABEL; // expected-error{{cannot jump from this goto statement to its label}}
  }

LABEL:
  {}

  goto LABEL_IN; // expected-error{{cannot jump from this goto statement to its label}}

#pragma acc parallel // expected-note{{invalid branch into OpenACC Compute Construct}}
  for(int i = 0; i < 5; ++i) {
LABEL_IN:
    {}
  }

#pragma acc parallel
  for(int i = 0; i < 5; ++i) {
LABEL_NOT_CALLED:
    {}
  }

#pragma acc parallel
  {
    goto ANOTHER_LOOP; // expected-error{{cannot jump from this goto statement to its label}}

  }
#pragma acc parallel// expected-note{{invalid branch into OpenACC Compute Construct}}

  {
ANOTHER_LOOP:
    {}
  }

#pragma acc parallel
  {
  while (j) {
    --j;
    if (j < 3)
      goto LABEL2;

    if (j > 4)
      break;
  }
LABEL2:
  {}
  }

#pragma acc parallel
  do {
    if (j < 3)
      goto LABEL3;

    if (j > 4)
      break; // expected-error{{invalid branch out of OpenACC Compute Construct}}

LABEL3:
  {}
  } while (j);

LABEL4:
  {}
#pragma acc parallel// expected-note{{invalid branch out of OpenACC Compute Construct}}
  {
    goto LABEL4;// expected-error{{cannot jump from this goto statement to its label}}
  }

#pragma acc parallel// expected-note{{invalid branch into OpenACC Compute Construct}}

  {
LABEL5:
    {}
  }

  {
    goto LABEL5;// expected-error{{cannot jump from this goto statement to its label}}
  }

#pragma acc parallel
  {
LABEL6:
    {}
    goto LABEL6;

  }

#pragma acc parallel
  goto LABEL7; // expected-error{{cannot jump from this goto statement to its label}}
#pragma acc parallel// expected-note{{invalid branch into OpenACC Compute Construct}}
  {
LABEL7:{}
  }

#pragma acc parallel
  LABEL8:{}
#pragma acc parallel// expected-note{{invalid branch out of OpenACC Compute Construct}}
  {
    goto LABEL8;// expected-error{{cannot jump from this goto statement to its label}}
  }


#pragma acc parallel// expected-note{{invalid branch into OpenACC Compute Construct}}
  {
LABEL9:{}
  }

  ({goto LABEL9;});// expected-error{{cannot jump from this goto statement to its label}}

#pragma acc parallel// expected-note{{invalid branch out of OpenACC Compute Construct}}
  {
  ({goto LABEL10;});// expected-error{{cannot jump from this goto statement to its label}}
  }

LABEL10:{}

  ({goto LABEL11;});// expected-error{{cannot jump from this goto statement to its label}}
#pragma acc parallel// expected-note{{invalid branch into OpenACC Compute Construct}}
  {
LABEL11:{}
  }

LABEL12:{}
#pragma acc parallel// expected-note{{invalid branch out of OpenACC Compute Construct}}
  {
  ({goto LABEL12;});// expected-error{{cannot jump from this goto statement to its label}}
  }

#pragma acc parallel
  {
  ({goto LABEL13;});
LABEL13:{}
  }

#pragma acc parallel
  {
  LABEL14:{}
  ({goto LABEL14;});
  }
}

void IndirectGoto1() {
  void* ptr;
#pragma acc parallel
  {
LABEL1:{}
    ptr = &&LABEL1;

    goto *ptr;

  }
}

void IndirectGoto2() {
  void* ptr;
LABEL2:{} // #GOTOLBL2
    ptr = &&LABEL2;
#pragma acc parallel // #GOTOPAR2
  {
// expected-error@+3{{cannot jump from this indirect goto statement to one of its possible targets}}
// expected-note@#GOTOLBL2{{possible target of indirect goto statement}}
// expected-note@#GOTOPAR2{{invalid branch out of OpenACC Compute Construct}}
    goto *ptr;
  }
}

void IndirectGoto3() {
  void* ptr;
#pragma acc parallel // #GOTOPAR3
  {
LABEL3:{} // #GOTOLBL3
    ptr = &&LABEL3;
  }
// expected-error@+3{{cannot jump from this indirect goto statement to one of its possible targets}}
// expected-note@#GOTOLBL3{{possible target of indirect goto statement}}
// expected-note@#GOTOPAR3{{invalid branch into OpenACC Compute Construct}}
  goto *ptr;
}

void IndirectGoto4() {
  void* ptr;
#pragma acc parallel // #GOTOPAR4
  {
LABEL4:{}
    ptr = &&LABEL4;
// expected-error@+3{{cannot jump from this indirect goto statement to one of its possible targets}}
// expected-note@#GOTOLBL5{{possible target of indirect goto statement}}
// expected-note@#GOTOPAR4{{invalid branch out of OpenACC Compute Construct}}
    goto *ptr;
  }
LABEL5:// #GOTOLBL5

  ptr=&&LABEL5;
}

void DuffsDevice() {
  int j;
  switch (j) {
#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    case 0: // expected-error{{invalid branch into OpenACC Compute Construct}}
      {}
  }
  }

  switch (j) {
#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    default: // expected-error{{invalid branch into OpenACC Compute Construct}}
      {}
  }
  }

  switch (j) {
#pragma acc parallel
  for(int i =0; i < 5; ++i) {
    case 'a' ... 'z': // expected-error{{invalid branch into OpenACC Compute Construct}}
      {}
  }
  }
}
