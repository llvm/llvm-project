#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bb10  ########


bb10: run
	

build:  $(SRC)/bb10.f
	-$(RM) bb10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/bb10.f -o bb10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bb10.$(OBJX) check.$(OBJX) $(LIBS) -o bb10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bb10
	bb10.$(EXESUFFIX)

verify: ;

bb10.run: run

