#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

build: $(SRC)/rio2.f90
	-$(FC) $(FFLAGS) $(LDFLAGS) -fopenmp $(SRC)/rio2.f90 -o rio2
run:
	-./rio2 
verify: ;
