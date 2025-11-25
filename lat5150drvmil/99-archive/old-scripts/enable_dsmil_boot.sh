#!/bin/bash
# Enable DSMIL AVX-512 driver on boot
echo "dsmil_avx512_enabler" > /etc/modules-load.d/dsmil-avx512.conf
echo "COMPLETE: DSMIL AVX-512 will auto-load on boot"
