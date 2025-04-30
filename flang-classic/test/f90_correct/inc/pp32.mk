#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp32  ########


pp32: run
	

build:  $(SRC)/pp32.f90
	-$(RM) pp32.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp32.f90 -o pp32.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp32.$(OBJX) check.$(OBJX) $(LIBS) -o pp32.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp32
	pp32.$(EXESUFFIX)

verify: ;

pp32.run: run

