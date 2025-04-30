#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
SRC2=$(SRC)/src
fs21390.$(OBJX): $(SRC2)/fs21390.f90
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC2)/fs21390.f90 -o fs21390.$(OBJX)
	-$(FC) $(LDFLAGS) fs21390.$(OBJX) -o fs21390.out

fs21390: fs21390.$(OBJX)
	-rm -f fs21390_run.log
	$(RUN4) fs21390.out 2>&1 > fs21390_run.log
	-$(NGREP) DEALLOCATE fs21390_run.log

build: fs21390

run: ;
