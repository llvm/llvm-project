#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bi01  ########


bi01: run
	

build:  $(SRC)/bi01.f90
	-$(RM) bi01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(CC) -c $(CFLAGS) $(SRC)/bi01.c -o bi01_c.$(OBJX)
	-$(FC) -c $(FFLAGS) $(SRC)/bi01.f90 -o bi01_f.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bi01_f.$(OBJX) bi01_c.$(OBJX) check.$(OBJX)  $(LIBS) -o bi01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bi01
	bi01.$(EXESUFFIX)

verify: ;

bi01.run: run

