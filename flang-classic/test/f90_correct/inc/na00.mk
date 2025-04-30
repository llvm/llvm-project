#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test na00  ########


na00: run
	

build:  $(SRC)/na00.f
	-$(RM) na00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/na00.f -o na00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) na00.$(OBJX) check.$(OBJX) $(LIBS) -o na00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test na00
	na00.$(EXESUFFIX)

verify: ;

na00.run: run

