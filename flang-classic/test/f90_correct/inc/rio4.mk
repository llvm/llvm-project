#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

build: $(SRC)/rio4.f90
	-$(FC) $(LDFLAGS) $(SRC)/rio4.f90 -o rio4
run:
	-./rio4
verify: ;
