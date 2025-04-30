#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test cb20  ########


cb20: run
	

build:  $(SRC)/cb20.f
	-$(RM) cb20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/cb20.f -o cb20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) cb20.$(OBJX) check.$(OBJX) $(LIBS) -o cb20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test cb20
	cb20.$(EXESUFFIX)

verify: ;

cb20.run: run

