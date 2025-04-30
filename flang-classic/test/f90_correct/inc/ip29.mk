#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip29  ########


ip29: run
	

build:  $(SRC)/ip29.f90
	-$(RM) ip29.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip29.f90 -o ip29.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip29.$(OBJX) check.$(OBJX) $(LIBS) -o ip29.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip29
	ip29.$(EXESUFFIX)

verify: ;

ip29.run: run

