; Simple bit of IR to mimic CUDA's libdevice.

target triple = "nvptx-unknown-cuda"

define double @__nv_sin(double %a) {
       ret double 1.0
}

define double @__nv_exp(double %a) {
       ret double 3.0
}

define double @__unused(double %a) {
       ret double 2.0
}

