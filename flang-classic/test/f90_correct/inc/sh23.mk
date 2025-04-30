#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test sh23  ########


sh23: run
	

build:  $(SRC)/sh23.f90
	-$(RM) sh23.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/sh23.f90 -o sh23.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) sh23.$(OBJX) check.$(OBJX) $(LIBS) -o sh23.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test sh23
	sh23.$(EXESUFFIX)

verify: ;

sh23.run: run

