#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
par07: par07.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
par07.$(OBJX): $(SRC)/par07.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/par07.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) par07.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: par07
run: ;
