#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ff03  ########


ff03: run
	

build:  $(SRC)/ff03.f90
	-$(RM) ff03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ff03.f90 -o ff03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ff03.$(OBJX) check.$(OBJX) $(LIBS) -o ff03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ff03
	ff03.$(EXESUFFIX)

verify: ;

ff03.run: run

