#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt27  ########


dt27: run
	

build:  $(SRC)/dt27.f90
	-$(RM) dt27.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt27.f90 -o dt27.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt27.$(OBJX) check.$(OBJX) $(LIBS) -o dt27.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt27
	dt27.$(EXESUFFIX)

verify: ;

dt27.run: run

