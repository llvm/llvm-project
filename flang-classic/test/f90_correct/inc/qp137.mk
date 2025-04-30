#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qpowk  ########


qp137: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp137.f08 fcheck.$(OBJX)
	-$(RM) qp137.$(EXESUFFIX) qp137.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp137.f08 -o qp137.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp137.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp137.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp137
	qp137.$(EXESUFFIX)

verify: ;

qp137.run: run

