#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test na20  ########


na20: run
	

build:  $(SRC)/na20.f
	-$(RM) na20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/na20.f -o na20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) na20.$(OBJX) check.$(OBJX) $(LIBS) -o na20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test na20
	na20.$(EXESUFFIX)

verify: ;

na20.run: run

