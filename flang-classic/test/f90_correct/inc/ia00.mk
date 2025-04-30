#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ia00  ########


ia00: run
	

build:  $(SRC)/ia00.f
	-$(RM) ia00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ia00.f -o ia00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ia00.$(OBJX) check.$(OBJX) $(LIBS) -o ia00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ia00
	ia00.$(EXESUFFIX)

verify: ;

ia00.run: run

