#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io  ########


qp37: run


build:  $(SRC)/qp37.f08
	-$(RM) qp37.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp37.f08 -o qp37.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp37.$(OBJX) check.$(OBJX) $(LIBS) -o qp37.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp37
	qp37.$(EXESUFFIX)

verify: ;


