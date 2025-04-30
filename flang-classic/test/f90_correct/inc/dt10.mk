#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dt10  ########


dt10: run
	

build:  $(SRC)/dt10.f90
	-$(RM) dt10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dt10.f90 -o dt10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dt10.$(OBJX) check.$(OBJX) $(LIBS) -o dt10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dt10
	dt10.$(EXESUFFIX)

verify: ;

dt10.run: run

