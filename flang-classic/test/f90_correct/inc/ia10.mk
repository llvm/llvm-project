#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ia10  ########


ia10: run
	

build:  $(SRC)/ia10.f
	-$(RM) ia10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ia10.f -o ia10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ia10.$(OBJX) check.$(OBJX) $(LIBS) -o ia10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ia10
	ia10.$(EXESUFFIX)

verify: ;

ia10.run: run

