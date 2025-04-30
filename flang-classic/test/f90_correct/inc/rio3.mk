#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

build: $(SRC)/rio3.f90
	-$(FC) $(FFLAGS) $(LDFLAGS) -fopenmp $(SRC)/rio3.f90 -o rio3
run:
	-./rio3 
verify: ;
