/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <hurd.h>
#include <hurd/msg_server.h>
#include <hurd/fd.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <argz.h>


#define AUTHCHECK \
  if (auth != mach_task_self () && ! __USEPORT (AUTH, port == auth)) \
    return EPERM


/* Snarfing and frobbing the init ports.  */

kern_return_t
  _S_msg_get_init_port (mach_port_t msgport, mach_port_t auth, int which,
			mach_port_t *result, mach_msg_type_name_t *result_type)
{
  AUTHCHECK;
  *result_type = MACH_MSG_TYPE_MOVE_SEND;
  /* This function adds a new user reference for the *RESULT it gives back.
     Our reply message uses a move-send right that consumes this reference.  */
  return _hurd_ports_get (which, result);
}

kern_return_t
_S_msg_set_init_port (mach_port_t msgport, mach_port_t auth,
		      int which, mach_port_t port)
{
  error_t err;

  AUTHCHECK;

  err = _hurd_ports_set (which, port);
  if (err == 0)
    __mach_port_deallocate (__mach_task_self (), port);

  return 0;
}

kern_return_t
_S_msg_get_init_ports (mach_port_t msgport, mach_port_t auth,
		       mach_port_t **ports,
		       mach_msg_type_name_t *ports_type,
		       mach_msg_type_number_t *nports)
{
  mach_msg_type_number_t i;
  error_t err;

  AUTHCHECK;

  if (err = __vm_allocate (__mach_task_self (), (vm_address_t *) ports,
			   _hurd_nports * sizeof (mach_port_t), 1))
    return err;
  *nports = _hurd_nports;

  for (i = 0; i < _hurd_nports; ++i)
    /* This function adds a new user ref for the *RESULT it gives back.
       Our reply message uses move-send rights that consumes this ref.  */
    if (err = _hurd_ports_get (i, &(*ports)[i]))
      {
	/* Died part way through.  Deallocate the ports already fetched.  */
	while (i-- > 0)
	  __mach_port_deallocate (__mach_task_self (), (*ports)[i]);
	__vm_deallocate (__mach_task_self (),
			 (vm_address_t) *ports,
			 *nports * sizeof (mach_port_t));
	return err;
      }

  *ports_type = MACH_MSG_TYPE_MOVE_SEND;
  return 0;
}

kern_return_t
_S_msg_set_init_ports (mach_port_t msgport, mach_port_t auth,
		       mach_port_t *ports, mach_msg_type_number_t nports)
{
  mach_msg_type_number_t i;
  error_t err;

  AUTHCHECK;

  for (i = 0; i < _hurd_nports; ++i)
    {
      if (err = _hurd_ports_set (i, ports[i]))
	return err;
      else
	__mach_port_deallocate (__mach_task_self (), ports[i]);
    }

  return 0;
}

/* Snarfing and frobbing the init ints.  */

static kern_return_t
get_int (int which, int *value)
{
  switch (which)
    {
    case INIT_UMASK:
      *value = _hurd_umask;
      return 0;
    case INIT_SIGPENDING:
      {
	struct hurd_sigstate *ss = _hurd_global_sigstate;
	__spin_lock (&ss->lock);
	*value = ss->pending;
	__spin_unlock (&ss->lock);
	return 0;
      }
    case INIT_SIGIGN:
      {
	struct hurd_sigstate *ss = _hurd_global_sigstate;
	sigset_t ign;
	int sig;
	__spin_lock (&ss->lock);
	__sigemptyset (&ign);
	for (sig = 1; sig < NSIG; ++sig)
	  if (ss->actions[sig].sa_handler == SIG_IGN)
	    __sigaddset (&ign, sig);
	__spin_unlock (&ss->lock);
	*value = ign;
	return 0;
      }
    default:
      return EINVAL;
    }
}

kern_return_t
_S_msg_get_init_int (mach_port_t msgport, mach_port_t auth,
		     int which, int *value)
{
  AUTHCHECK;

  return get_int (which, value);
}

kern_return_t
_S_msg_get_init_ints (mach_port_t msgport, mach_port_t auth,
		      int **values, mach_msg_type_number_t *nvalues)
{
  error_t err;
  mach_msg_type_number_t i;

  AUTHCHECK;

  if (err = __vm_allocate (__mach_task_self (), (vm_address_t *) values,
			   INIT_INT_MAX * sizeof (int), 1))
    return err;
  *nvalues = INIT_INT_MAX;

  for (i = 0; i < INIT_INT_MAX; ++i)
    switch (err = get_int (i, &(*values)[i]))
      {
      case 0:			/* Success.  */
	break;
      case EINVAL:		/* Unknown index.  */
	(*values)[i] = 0;
	break;
      default:			/* Lossage.  */
	__vm_deallocate (__mach_task_self (),
			 (vm_address_t) *values, INIT_INT_MAX * sizeof (int));
	return err;
      }

  return 0;
}


static kern_return_t
set_int (int which, int value)
{
  switch (which)
    {
    case INIT_UMASK:
      _hurd_umask = value;
      return 0;

      /* These are pretty odd things to do.  But you asked for it.  */
    case INIT_SIGPENDING:
      {
	struct hurd_sigstate *ss = _hurd_global_sigstate;
	__spin_lock (&ss->lock);
	ss->pending = value;
	__spin_unlock (&ss->lock);
	return 0;
      }
    case INIT_SIGIGN:
      {
	struct hurd_sigstate *ss = _hurd_global_sigstate;
	int sig;
	const sigset_t ign = value;
	__spin_lock (&ss->lock);
	for (sig = 1; sig < NSIG; ++sig)
	  {
	    if (__sigismember (&ign, sig))
	      ss->actions[sig].sa_handler = SIG_IGN;
	    else if (ss->actions[sig].sa_handler == SIG_IGN)
	      ss->actions[sig].sa_handler = SIG_DFL;
	  }
	__spin_unlock (&ss->lock);
	return 0;

      case INIT_TRACEMASK:
	_hurdsig_traced = value;
	return 0;
      }
    default:
      return EINVAL;
    }
}

kern_return_t
_S_msg_set_init_int (mach_port_t msgport, mach_port_t auth,
		     int which, int value)
{
  AUTHCHECK;

  return set_int (which, value);
}

kern_return_t
_S_msg_set_init_ints (mach_port_t msgport, mach_port_t auth,
		      int *values, mach_msg_type_number_t nvalues)
{
  error_t err;
  mach_msg_type_number_t i;

  AUTHCHECK;

  for (i = 0; i < INIT_INT_MAX; ++i)
    switch (err = set_int (i, values[i]))
      {
      case 0:			/* Success.  */
	break;
      case EINVAL:		/* Unknown index.  */
	break;
      default:			/* Lossage.  */
	return err;
      }

  return 0;
}


kern_return_t
_S_msg_get_fd (mach_port_t msgport, mach_port_t auth, int which,
	       mach_port_t *result, mach_msg_type_name_t *result_type)
{
  AUTHCHECK;

  /* This creates a new user reference for the send right.
     Our reply message will move that reference to the caller.  */
  *result = __getdport (which);
  if (*result == MACH_PORT_NULL)
    return errno;
  *result_type = MACH_MSG_TYPE_MOVE_SEND;

  return 0;
}

kern_return_t
_S_msg_set_fd (mach_port_t msgport, mach_port_t auth,
	       int which, mach_port_t port)
{
  AUTHCHECK;

  /* We consume the reference if successful.  */
  return HURD_FD_USE (which, (_hurd_port2fd (descriptor, port, 0), 0));
}

/* Snarfing and frobbing environment variables.  */

kern_return_t
_S_msg_get_env_variable (mach_port_t msgport,
			 string_t variable, //
			 char **data, mach_msg_type_number_t *datalen)
{
  error_t err;
  mach_msg_type_number_t valuelen;
  const char *value = getenv (variable);

  if (value == NULL)
    return ENOENT;

  valuelen = strlen (value);
  if (valuelen > *datalen)
    {
      if (err = __vm_allocate (__mach_task_self (),
			       (vm_address_t *) data, valuelen, 1))
	return err;
    }

  memcpy (*data, value, valuelen);
  *datalen = valuelen;

  return 0;
}


kern_return_t
_S_msg_set_env_variable (mach_port_t msgport, mach_port_t auth,
			 string_t variable, //
			 string_t value, //
			 int replace)
{
  AUTHCHECK;

  if (__setenv (variable, value, replace)) /* XXX name space */
    return errno;
  return 0;
}

kern_return_t
_S_msg_get_environment (mach_port_t msgport,
			char **data, mach_msg_type_number_t *datalen)
{
  /* Pack the environment into an array with nulls separating elements.  */
  if (__environ != NULL)
    {
      char *ap, **p;
      size_t envlen = 0;

      for (p = __environ; *p != NULL; ++p)
	envlen += strlen (*p) + 1;

      if (envlen > *datalen)
	{
	  if (__vm_allocate (__mach_task_self (),
			     (vm_address_t *) data, envlen, 1))
	    return ENOMEM;
	}

      ap = *data;
      for (p = __environ; *p != NULL; ++p)
	ap = __memccpy (ap, *p, '\0', ULONG_MAX);

      *datalen = envlen;
    }
  else
    *datalen = 0;

  return 0;
}

kern_return_t
_S_msg_set_environment (mach_port_t msgport, mach_port_t auth,
			char *data, mach_msg_type_number_t datalen)
{
  int _hurd_split_args (char *, mach_msg_type_number_t, char **);
  int envc;
  char **envp;

  AUTHCHECK;

  envc = __argz_count (data, datalen);
  envp = malloc ((envc + 1) * sizeof (char *));
  if (envp == NULL)
    return errno;
  __argz_extract (data, datalen, envp);
  __environ = envp;		/* XXX cooperate with loadenv et al */
  return 0;
}


/* XXX */

kern_return_t
_S_msg_get_dtable (mach_port_t process,
		   mach_port_t refport,
		   portarray_t *dtable,
		   mach_msg_type_name_t *dtablePoly,
		   mach_msg_type_number_t *dtableCnt)
{ return EOPNOTSUPP; }

kern_return_t
_S_msg_set_dtable (mach_port_t process,
		   mach_port_t refport,
		   portarray_t dtable,
		   mach_msg_type_number_t dtableCnt)
{ return EOPNOTSUPP; }
