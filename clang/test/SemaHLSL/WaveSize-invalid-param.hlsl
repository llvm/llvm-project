// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -x hlsl %s -verify

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wavesize arguments must be between 4 and 128 and a power of 2}}
[WaveSize(1)]
void e0() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wavesize arguments must be between 4 and 128 and a power of 2}}
[WaveSize(4, 2)]
void e1() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wavesize arguments must be between 4 and 128 and a power of 2}}
[WaveSize(4, 8, 7)]
void e2() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{minimum wavesize value 16 must be less than maximum wavesize value 8}}
[WaveSize(16, 8)]
void e3() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{preferred wavesize value 8 must be between 16 and 128}}
[WaveSize(16, 128, 8)]
void e4() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{preferred wavesize value 32 must be between 8 and 16}}
[WaveSize(8, 16, 32)]
void e5() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wavesize arguments must be between 4 and 128 and a power of 2}}
[WaveSize(4, 0)]
void e6() {
}


[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wavesize arguments must be between 4 and 128 and a power of 2}}
[WaveSize(4, 4, 0)]
void e7() {
}


[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wave size range minimum and maximum are equal}}
[WaveSize(16, 16)]
void e8() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wavesize arguments must be between 4 and 128 and a power of 2}}
[WaveSize(0)]
void e9() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{wavesize arguments must be between 4 and 128 and a power of 2}}
[WaveSize(-4)]
void e10() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{'WaveSize' attribute takes no more than 3 arguments}}
[WaveSize(16, 128, 64, 64)]
void e11() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{'WaveSize' attribute takes at least 1 argument}}
[WaveSize()]
void e12() {
}

[shader("compute")]
[numthreads(1,1,1)]
// expected-error@+1 {{'WaveSize' attribute takes at least 1 argument}}
[WaveSize]
void e13() {
}
