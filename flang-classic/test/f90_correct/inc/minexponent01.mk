#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test minexponent01  ########


minexponent01: run
	

build:  $(SRC)/minexponent01.f08
	-$(RM) minexponent01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/minexponent01.f08 -Mpreprocess -o minexponent01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) minexponent01.$(OBJX) check.$(OBJX) $(LIBS) -o minexponent01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test minexponent01
	minexponent01.$(EXESUFFIX)

verify: ;

minexponent01.run: run

