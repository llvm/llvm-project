#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp33  ########


qp33: run


build:  $(SRC)/qp33.f08
	-$(RM) qp33.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp33.f08 -o qp33.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp33.$(OBJX) check.$(OBJX) $(LIBS) -o qp33.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp33
	qp33.$(EXESUFFIX)

verify: ;
