from __future__ import absolute_import

# System modules
import curses
import curses.panel
import sys
import time

# Third-party modules
import six

# LLDB modules


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(x=%u, y=%u)" % (self.x, self.y)

    def __eq__(self, rhs):
        return self.x == rhs.x and self.y == rhs.y

    def __ne__(self, rhs):
        return self.x != rhs.x or self.y != rhs.y

    def is_valid_coordinate(self):
        return self.x >= 0 and self.y >= 0


class Size(object):

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(w=%u, h=%u)" % (self.w, self.h)

    def __eq__(self, rhs):
        return self.w == rhs.w and self.h == rhs.h

    def __ne__(self, rhs):
        return self.w != rhs.w or self.h != rhs.h


class Rect(object):

    def __init__(self, x=0, y=0, w=0, h=0):
        self.origin = Point(x, y)
        self.size = Size(w, h)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "{ %s, %s }" % (str(self.origin), str(self.size))

    def get_min_x(self):
        return self.origin.x

    def get_max_x(self):
        return self.origin.x + self.size.w

    def get_min_y(self):
        return self.origin.y

    def get_max_y(self):
        return self.origin.y + self.size.h

    def contains_point(self, pt):
        if pt.x < self.get_max_x():
            if pt.y < self.get_max_y():
                if pt.x >= self.get_min_y():
                    return pt.y >= self.get_min_y()
        return False

    def __eq__(self, rhs):
        return self.origin == rhs.origin and self.size == rhs.size

    def __ne__(self, rhs):
        return self.origin != rhs.origin or self.size != rhs.size


class QuitException(Exception):

    def __init__(self):
        super(QuitException, self).__init__('QuitException')


class Window(object):

    def __init__(self, window, delegate=None, can_become_first_responder=True):
        self.window = window
        self.parent = None
        self.delegate = delegate
        self.children = list()
        self.first_responders = list()
        self.can_become_first_responder = can_become_first_responder
        self.key_actions = dict()

    def add_child(self, window):
        self.children.append(window)
        window.parent = self

    def resize(self, size):
        self.window.resize(size.h, size.w)

    def resize_child(self, child, delta_size, adjust_neighbors):
        if child in self.children:
            frame = self.get_frame()
            orig_frame = child.get_frame()
            new_frame = Rect(
                x=orig_frame.origin.x,
                y=orig_frame.origin.y,
                w=orig_frame.size.w +
                delta_size.w,
                h=orig_frame.size.h +
                delta_size.h)
            old_child_max_x = orig_frame.get_max_x()
            new_child_max_x = new_frame.get_max_x()
            window_max_x = frame.get_max_x()
            if new_child_max_x < window_max_x:
                child.resize(new_frame.size)
                if old_child_max_x == window_max_x:
                    new_frame.origin.x += window_max_x - new_child_max_x
                    child.set_position(new_frame.origin)
            elif new_child_max_x > window_max_x:
                new_frame.origin.x -= new_child_max_x - window_max_x
                child.set_position(new_frame.origin)
                child.resize(new_frame.size)

            if adjust_neighbors:
                #print('orig_frame = %s\r\n' % (str(orig_frame)), end='')
                for curr_child in self.children:
                    if curr_child is child:
                        continue
                    curr_child_frame = curr_child.get_frame()
                    if delta_size.w != 0:
                        #print('curr_child_frame = %s\r\n' % (str(curr_child_frame)), end='')
                        if curr_child_frame.get_min_x() == orig_frame.get_max_x():
                            curr_child_frame.origin.x += delta_size.w
                            curr_child_frame.size.w -= delta_size.w
                            #print('adjusted curr_child_frame = %s\r\n' % (str(curr_child_frame)), end='')
                            curr_child.resize(curr_child_frame.size)
                            curr_child.slide_position(
                                Size(w=delta_size.w, h=0))
                        elif curr_child_frame.get_max_x() == orig_frame.get_min_x():
                            curr_child_frame.size.w -= delta_size.w
                            #print('adjusted curr_child_frame = %s\r\n' % (str(curr_child_frame)), end='')
                            curr_child.resize(curr_child_frame.size)

    def add_key_action(self, arg, callback, decription):
        if isinstance(arg, list):
            for key in arg:
                self.add_key_action(key, callback, description)
        else:
            if isinstance(arg, six.integer_types):
                key_action_dict = {'key': arg,
                                   'callback': callback,
                                   'description': decription}
                self.key_actions[arg] = key_action_dict
            elif isinstance(arg, basestring):
                key_integer = ord(arg)
                key_action_dict = {'key': key_integer,
                                   'callback': callback,
                                   'description': decription}
                self.key_actions[key_integer] = key_action_dict
            else:
                raise ValueError

    def draw_title_box(self, title):
        is_in_first_responder_chain = self.is_in_first_responder_chain()
        if is_in_first_responder_chain:
            self.attron(curses.A_REVERSE)
        self.box()
        if is_in_first_responder_chain:
            self.attroff(curses.A_REVERSE)
        if title:
            self.addstr(Point(x=2, y=0), ' ' + title + ' ')

    def remove_child(self, window):
        self.children.remove(window)

    def get_first_responder(self):
        if len(self.first_responders):
            return self.first_responders[-1]
        else:
            return None

    def set_first_responder(self, window):
        if window.can_become_first_responder:
            if six.callable(
                getattr(
                    window,
                    "hidden",
                    None)) and window.hidden():
                return False
            if window not in self.children:
                self.add_child(window)
            # See if we have a current first responder, and if we do, let it know that
            # it will be resigning as first responder
            first_responder = self.get_first_responder()
            if first_responder:
                first_responder.relinquish_first_responder()
            # Now set the first responder to "window"
            if len(self.first_responders) == 0:
                self.first_responders.append(window)
            else:
                self.first_responders[-1] = window
            return True
        else:
            return False

    def push_first_responder(self, window):
        # Only push the window as the new first responder if the window isn't
        # already the first responder
        if window != self.get_first_responder():
            self.first_responders.append(window)

    def pop_first_responder(self, window):
        # Only pop the window from the first responder list if it is the first
        # responder
        if window == self.get_first_responder():
            old_first_responder = self.first_responders.pop()
            old_first_responder.relinquish_first_responder()
            return True
        else:
            return False

    def relinquish_first_responder(self):
        '''Override if there is something that you need to do when you lose first responder status.'''
        pass

    # def resign_first_responder(self, remove_from_parent, new_first_responder):
    #     success = False
    #     if self.parent:
    #         if self.is_first_responder():
    #             self.relinquish_first_responder()
    #             if len(self.parent.first_responder):
    #             self.parent.first_responder = None
    #             success = True
    #         if remove_from_parent:
    #             self.parent.remove_child(self)
    #         if new_first_responder:
    #             self.parent.set_first_responder(new_first_responder)
    #         else:
    #             self.parent.select_next_first_responder()
    #     return success

    def is_first_responder(self):
        if self.parent:
            return self.parent.get_first_responder() == self
        else:
            return False

    def is_in_first_responder_chain(self):
        if self.parent:
            return self in self.parent.first_responders
        else:
            return False

    def select_next_first_responder(self):
        if len(self.first_responders) > 1:
            self.pop_first_responder(self.first_responders[-1])
        else:
            num_children = len(self.children)
            if num_children == 1:
                return self.set_first_responder(self.children[0])
            for (i, window) in enumerate(self.children):
                if window.is_first_responder():
                    break
            if i < num_children:
                for i in range(i + 1, num_children):
                    if self.set_first_responder(self.children[i]):
                        return True
                for i in range(0, i):
                    if self.set_first_responder(self.children[i]):
                        return True

    def point_in_window(self, pt):
        size = self.get_size()
        return pt.x >= 0 and pt.x < size.w and pt.y >= 0 and pt.y < size.h

    def addch(self, c):
        try:
            self.window.addch(c)
        except:
            pass

    def addch_at_point(self, pt, c):
        try:
            self.window.addch(pt.y, pt.x, c)
        except:
            pass

    def addstr(self, pt, str):
        try:
            self.window.addstr(pt.y, pt.x, str)
        except:
            pass

    def addnstr_at_point(self, pt, str, n):
        try:
            self.window.addnstr(pt.y, pt.x, str, n)
        except:
            pass

    def addnstr(self, str, n):
        try:
            self.window.addnstr(str, n)
        except:
            pass

    def attron(self, attr):
        return self.window.attron(attr)

    def attroff(self, attr):
        return self.window.attroff(attr)

    def box(self, vertch=0, horch=0):
        if vertch == 0:
            vertch = curses.ACS_VLINE
        if horch == 0:
            horch = curses.ACS_HLINE
        self.window.box(vertch, horch)

    def get_contained_rect(
            self,
            top_inset=0,
            bottom_inset=0,
            left_inset=0,
            right_inset=0,
            height=-1,
            width=-1):
        '''Get a rectangle based on the top "height" lines of this window'''
        rect = self.get_frame()
        x = rect.origin.x + left_inset
        y = rect.origin.y + top_inset
        if height == -1:
            h = rect.size.h - (top_inset + bottom_inset)
        else:
            h = height
        if width == -1:
            w = rect.size.w - (left_inset + right_inset)
        else:
            w = width
        return Rect(x=x, y=y, w=w, h=h)

    def erase(self):
        self.window.erase()

    def get_cursor(self):
        (y, x) = self.window.getyx()
        return Point(x=x, y=y)

    def get_frame(self):
        position = self.get_position()
        size = self.get_size()
        return Rect(x=position.x, y=position.y, w=size.w, h=size.h)

    def get_frame_in_parent(self):
        position = self.get_position_in_parent()
        size = self.get_size()
        return Rect(x=position.x, y=position.y, w=size.w, h=size.h)

    def get_position_in_parent(self):
        (y, x) = self.window.getparyx()
        return Point(x, y)

    def get_position(self):
        (y, x) = self.window.getbegyx()
        return Point(x, y)

    def get_size(self):
        (y, x) = self.window.getmaxyx()
        return Size(w=x, h=y)

    def move(self, pt):
        self.window.move(pt.y, pt.x)

    def refresh(self):
        self.update()
        curses.panel.update_panels()
        self.move(Point(x=0, y=0))
        return self.window.refresh()

    def resize(self, size):
        return self.window.resize(size.h, size.w)

    def timeout(self, timeout_msec):
        return self.window.timeout(timeout_msec)

    def handle_key(self, key, check_parent=True):
        '''Handle a key press in this window.'''

        # First try the first responder if this window has one, but don't allow
        # it to check with its parent (False second parameter) so we don't recurse
        # and get a stack overflow
        for first_responder in reversed(self.first_responders):
            if first_responder.handle_key(key, False):
                return True

        # Check our key map to see if we have any actions. Actions don't take
        # any arguments, they must be callable
        if key in self.key_actions:
            key_action = self.key_actions[key]
            key_action['callback']()
            return True
        # Check if there is a wildcard key for any key
        if -1 in self.key_actions:
            key_action = self.key_actions[-1]
            key_action['callback']()
            return True
        # Check if the window delegate wants to handle this key press
        if self.delegate:
            if six.callable(getattr(self.delegate, "handle_key", None)):
                if self.delegate.handle_key(self, key):
                    return True
            if self.delegate(self, key):
                return True
        # Check if we have a parent window and if so, let the parent
        # window handle the key press
        if check_parent and self.parent:
            return self.parent.handle_key(key, True)
        else:
            return False  # Key not handled

    def update(self):
        for child in self.children:
            child.update()

    def quit_action(self):
        raise QuitException

    def get_key(self, timeout_msec=-1):
        self.timeout(timeout_msec)
        done = False
        c = self.window.getch()
        if c == 27:
            self.timeout(0)
            escape_key = 0
            while True:
                escape_key = self.window.getch()
                if escape_key == -1:
                    break
                else:
                    c = c << 8 | escape_key
            self.timeout(timeout_msec)
        return c

    def key_event_loop(self, timeout_msec=-1, n=sys.maxsize):
        '''Run an event loop to receive key presses and pass them along to the
           responder chain.

           timeout_msec is the timeout it milliseconds. If the value is -1, an
           infinite wait will be used. It the value is zero, a non-blocking mode
           will be used, and if greater than zero it will wait for a key press
           for timeout_msec milliseconds.

           n is the number of times to go through the event loop before exiting'''
        done = False
        while not done and n > 0:
            c = self.get_key(timeout_msec)
            if c != -1:
                try:
                    self.handle_key(c)
                except QuitException:
                    done = True
            n -= 1


class Panel(Window):

    def __init__(self, frame, delegate=None, can_become_first_responder=True):
        window = curses.newwin(
            frame.size.h,
            frame.size.w,
            frame.origin.y,
            frame.origin.x)
        super(
            Panel,
            self).__init__(
            window,
            delegate,
            can_become_first_responder)
        self.panel = curses.panel.new_panel(window)

    def hide(self):
        return self.panel.hide()

    def hidden(self):
        return self.panel.hidden()

    def show(self):
        return self.panel.show()

    def top(self):
        return self.panel.top()

    def set_position(self, pt):
        self.panel.move(pt.y, pt.x)

    def slide_position(self, size):
        new_position = self.get_position()
        new_position.x = new_position.x + size.w
        new_position.y = new_position.y + size.h
        self.set_position(new_position)


class BoxedPanel(Panel):

    def __init__(self, frame, title, delegate=None,
                 can_become_first_responder=True):
        super(
            BoxedPanel,
            self).__init__(
            frame,
            delegate,
            can_become_first_responder)
        self.title = title
        self.lines = list()
        self.first_visible_idx = 0
        self.selected_idx = -1
        self.add_key_action(
            curses.KEY_UP,
            self.select_prev,
            "Select the previous item")
        self.add_key_action(
            curses.KEY_DOWN,
            self.select_next,
            "Select the next item")
        self.add_key_action(
            curses.KEY_HOME,
            self.scroll_begin,
            "Go to the beginning of the list")
        self.add_key_action(
            curses.KEY_END,
            self.scroll_end,
            "Go to the end of the list")
        self.add_key_action(
            0x1b4f48,
            self.scroll_begin,
            "Go to the beginning of the list")
        self.add_key_action(
            0x1b4f46,
            self.scroll_end,
            "Go to the end of the list")
        self.add_key_action(
            curses.KEY_PPAGE,
            self.scroll_page_backward,
            "Scroll to previous page")
        self.add_key_action(
            curses.KEY_NPAGE,
            self.scroll_page_forward,
            "Scroll to next forward")
        self.update()

    def clear(self, update=True):
        self.lines = list()
        self.first_visible_idx = 0
        self.selected_idx = -1
        if update:
            self.update()

    def get_usable_width(self):
        '''Valid usable width is 0 to (width - 3) since the left and right lines display the box around
           this frame and we skip a leading space'''
        w = self.get_size().w
        if w > 3:
            return w - 3
        else:
            return 0

    def get_usable_height(self):
        '''Valid line indexes are 0 to (height - 2) since the top and bottom lines display the box around this frame.'''
        h = self.get_size().h
        if h > 2:
            return h - 2
        else:
            return 0

    def get_point_for_line(self, global_line_idx):
        '''Returns the point to use when displaying a line whose index is "line_idx"'''
        line_idx = global_line_idx - self.first_visible_idx
        num_lines = self.get_usable_height()
        if line_idx < num_lines:
            return Point(x=2, y=1 + line_idx)
        else:
            # return an invalid coordinate if the line index isn't valid
            return Point(x=-1, y=-1)

    def set_title(self, title, update=True):
        self.title = title
        if update:
            self.update()

    def scroll_to_line(self, idx):
        if idx < len(self.lines):
            self.selected_idx = idx
            max_visible_lines = self.get_usable_height()
            if idx < self.first_visible_idx or idx >= self.first_visible_idx + max_visible_lines:
                self.first_visible_idx = idx
            self.refresh()

    def scroll_begin(self):
        self.first_visible_idx = 0
        if len(self.lines) > 0:
            self.selected_idx = 0
        else:
            self.selected_idx = -1
        self.update()

    def scroll_end(self):
        max_visible_lines = self.get_usable_height()
        num_lines = len(self.lines)
        if num_lines > max_visible_lines:
            self.first_visible_idx = num_lines - max_visible_lines
        else:
            self.first_visible_idx = 0
        self.selected_idx = num_lines - 1
        self.update()

    def scroll_page_backward(self):
        num_lines = len(self.lines)
        max_visible_lines = self.get_usable_height()
        new_index = self.first_visible_idx - max_visible_lines
        if new_index < 0:
            self.first_visible_idx = 0
        else:
            self.first_visible_idx = new_index
        self.refresh()

    def scroll_page_forward(self):
        max_visible_lines = self.get_usable_height()
        self.first_visible_idx += max_visible_lines
        self._adjust_first_visible_line()
        self.refresh()

    def select_next(self):
        self.selected_idx += 1
        if self.selected_idx >= len(self.lines):
            self.selected_idx = len(self.lines) - 1
        self.refresh()

    def select_prev(self):
        self.selected_idx -= 1
        if self.selected_idx < 0:
            if len(self.lines) > 0:
                self.selected_idx = 0
            else:
                self.selected_idx = -1
        self.refresh()

    def get_selected_idx(self):
        return self.selected_idx

    def _adjust_first_visible_line(self):
        num_lines = len(self.lines)
        max_visible_lines = self.get_usable_height()
        if (self.first_visible_idx >= num_lines) or (
                num_lines - self.first_visible_idx) > max_visible_lines:
            self.first_visible_idx = num_lines - max_visible_lines

    def append_line(self, s, update=True):
        self.lines.append(s)
        self._adjust_first_visible_line()
        if update:
            self.update()

    def set_line(self, line_idx, s, update=True):
        '''Sets a line "line_idx" within the boxed panel to be "s"'''
        if line_idx < 0:
            return
        while line_idx >= len(self.lines):
            self.lines.append('')
        self.lines[line_idx] = s
        self._adjust_first_visible_line()
        if update:
            self.update()

    def update(self):
        self.erase()
        self.draw_title_box(self.title)
        max_width = self.get_usable_width()
        for line_idx in range(self.first_visible_idx, len(self.lines)):
            pt = self.get_point_for_line(line_idx)
            if pt.is_valid_coordinate():
                is_selected = line_idx == self.selected_idx
                if is_selected:
                    self.attron(curses.A_REVERSE)
                self.move(pt)
                self.addnstr(self.lines[line_idx], max_width)
                if is_selected:
                    self.attroff(curses.A_REVERSE)
            else:
                return

    def load_file(self, path):
        f = open(path)
        if f:
            self.lines = f.read().splitlines()
            for (idx, line) in enumerate(self.lines):
                # Remove any tabs from lines since they hose up the display
                if "\t" in line:
                    self.lines[idx] = (8 * ' ').join(line.split('\t'))
        self.selected_idx = 0
        self.first_visible_idx = 0
        self.refresh()


class Item(object):

    def __init__(self, title, action):
        self.title = title
        self.action = action


class TreeItemDelegate(object):

    def might_have_children(self):
        return False

    def update_children(self, item):
        '''Return a list of child Item objects'''
        return None

    def draw_item_string(self, tree_window, item, s):
        pt = tree_window.get_cursor()
        width = tree_window.get_size().w - 1
        if width > pt.x:
            tree_window.addnstr(s, width - pt.x)

    def draw_item(self, tree_window, item):
        self.draw_item_string(tree_window, item, item.title)

    def do_action(self):
        pass


class TreeItem(object):

    def __init__(
            self,
            delegate,
            parent=None,
            title=None,
            action=None,
            is_expanded=False):
        self.parent = parent
        self.title = title
        self.action = action
        self.delegate = delegate
        self.is_expanded = not parent or is_expanded
        self._might_have_children = None
        self.children = None
        self._children_might_have_children = False

    def get_children(self):
        if self.is_expanded and self.might_have_children():
            if self.children is None:
                self._children_might_have_children = False
                self.children = self.update_children()
                for child in self.children:
                    if child.might_have_children():
                        self._children_might_have_children = True
                        break
        else:
            self._children_might_have_children = False
            self.children = None
        return self.children

    def append_visible_items(self, items):
        items.append(self)
        children = self.get_children()
        if children:
            for child in children:
                child.append_visible_items(items)

    def might_have_children(self):
        if self._might_have_children is None:
            if not self.parent:
                # Root item always might have children
                self._might_have_children = True
            else:
                # Check with the delegate to see if the item might have
                # children
                self._might_have_children = self.delegate.might_have_children()
        return self._might_have_children

    def children_might_have_children(self):
        return self._children_might_have_children

    def update_children(self):
        if self.is_expanded and self.might_have_children():
            self.children = self.delegate.update_children(self)
            for child in self.children:
                child.update_children()
        else:
            self.children = None
        return self.children

    def get_num_visible_rows(self):
        rows = 1
        if self.is_expanded:
            children = self.get_children()
            if children:
                for child in children:
                    rows += child.get_num_visible_rows()
        return rows

    def draw(self, tree_window, row):
        display_row = tree_window.get_display_row(row)
        if display_row >= 0:
            tree_window.move(tree_window.get_item_draw_point(row))
            if self.parent:
                self.parent.draw_tree_for_child(tree_window, self, 0)
            if self.might_have_children():
                tree_window.addch(curses.ACS_DIAMOND)
                tree_window.addch(curses.ACS_HLINE)
            elif self.parent and self.parent.children_might_have_children():
                if self.parent.parent:
                    tree_window.addch(curses.ACS_HLINE)
                    tree_window.addch(curses.ACS_HLINE)
                else:
                    tree_window.addch(' ')
                    tree_window.addch(' ')
            is_selected = tree_window.is_selected(row)
            if is_selected:
                tree_window.attron(curses.A_REVERSE)
            self.delegate.draw_item(tree_window, self)
            if is_selected:
                tree_window.attroff(curses.A_REVERSE)

    def draw_tree_for_child(self, tree_window, child, reverse_depth):
        if self.parent:
            self.parent.draw_tree_for_child(
                tree_window, self, reverse_depth + 1)
            if self.children[-1] == child:
                # Last child
                if reverse_depth == 0:
                    tree_window.addch(curses.ACS_LLCORNER)
                    tree_window.addch(curses.ACS_HLINE)
                else:
                    tree_window.addch(' ')
                    tree_window.addch(' ')
            else:
                # Middle child
                if reverse_depth == 0:
                    tree_window.addch(curses.ACS_LTEE)
                    tree_window.addch(curses.ACS_HLINE)
                else:
                    tree_window.addch(curses.ACS_VLINE)
                    tree_window.addch(' ')

    def was_selected(self):
        self.delegate.do_action()


class TreePanel(Panel):

    def __init__(self, frame, title, root_item):
        self.root_item = root_item
        self.title = title
        self.first_visible_idx = 0
        self.selected_idx = 0
        self.items = None
        super(TreePanel, self).__init__(frame)
        self.add_key_action(
            curses.KEY_UP,
            self.select_prev,
            "Select the previous item")
        self.add_key_action(
            curses.KEY_DOWN,
            self.select_next,
            "Select the next item")
        self.add_key_action(
            curses.KEY_RIGHT,
            self.right_arrow,
            "Expand an item")
        self.add_key_action(
            curses.KEY_LEFT,
            self.left_arrow,
            "Unexpand an item or navigate to parent")
        self.add_key_action(
            curses.KEY_HOME,
            self.scroll_begin,
            "Go to the beginning of the tree")
        self.add_key_action(
            curses.KEY_END,
            self.scroll_end,
            "Go to the end of the tree")
        self.add_key_action(
            0x1b4f48,
            self.scroll_begin,
            "Go to the beginning of the tree")
        self.add_key_action(
            0x1b4f46,
            self.scroll_end,
            "Go to the end of the tree")
        self.add_key_action(
            curses.KEY_PPAGE,
            self.scroll_page_backward,
            "Scroll to previous page")
        self.add_key_action(
            curses.KEY_NPAGE,
            self.scroll_page_forward,
            "Scroll to next forward")

    def get_selected_item(self):
        if self.selected_idx < len(self.items):
            return self.items[self.selected_idx]
        else:
            return None

    def select_item(self, item):
        if self.items and item in self.items:
            self.selected_idx = self.items.index(item)
            return True
        else:
            return False

    def get_visible_items(self):
        # Clear self.items when you want to update all chidren
        if self.items is None:
            self.items = list()
            children = self.root_item.get_children()
            if children:
                for child in children:
                    child.append_visible_items(self.items)
        return self.items

    def update(self):
        self.erase()
        self.draw_title_box(self.title)
        visible_items = self.get_visible_items()
        for (row, child) in enumerate(visible_items):
            child.draw(self, row)

    def get_item_draw_point(self, row):
        display_row = self.get_display_row(row)
        if display_row >= 0:
            return Point(2, display_row + 1)
        else:
            return Point(-1, -1)

    def get_display_row(self, row):
        if row >= self.first_visible_idx:
            display_row = row - self.first_visible_idx
            if display_row < self.get_size().h - 2:
                return display_row
        return -1

    def is_selected(self, row):
        return row == self.selected_idx

    def get_num_lines(self):
        self.get_visible_items()
        return len(self.items)

    def get_num_visible_lines(self):
        return self.get_size().h - 2

    def select_next(self):
        self.selected_idx += 1
        num_lines = self.get_num_lines()
        if self.selected_idx >= num_lines:
            self.selected_idx = num_lines - 1
        self._selection_changed()
        self.refresh()

    def select_prev(self):
        self.selected_idx -= 1
        if self.selected_idx < 0:
            num_lines = self.get_num_lines()
            if num_lines > 0:
                self.selected_idx = 0
            else:
                self.selected_idx = -1
        self._selection_changed()
        self.refresh()

    def scroll_begin(self):
        self.first_visible_idx = 0
        num_lines = self.get_num_lines()
        if num_lines > 0:
            self.selected_idx = 0
        else:
            self.selected_idx = -1
        self.refresh()

    def redisplay_tree(self):
        self.items = None
        self.refresh()

    def right_arrow(self):
        selected_item = self.get_selected_item()
        if selected_item and selected_item.is_expanded == False:
            selected_item.is_expanded = True
            self.redisplay_tree()

    def left_arrow(self):
        selected_item = self.get_selected_item()
        if selected_item:
            if selected_item.is_expanded:
                selected_item.is_expanded = False
                self.redisplay_tree()
            elif selected_item.parent:
                if self.select_item(selected_item.parent):
                    self.refresh()

    def scroll_end(self):
        num_visible_lines = self.get_num_visible_lines()
        num_lines = self.get_num_lines()
        if num_lines > num_visible_lines:
            self.first_visible_idx = num_lines - num_visible_lines
        else:
            self.first_visible_idx = 0
        self.selected_idx = num_lines - 1
        self.refresh()

    def scroll_page_backward(self):
        num_visible_lines = self.get_num_visible_lines()
        new_index = self.selected_idx - num_visible_lines
        if new_index < 0:
            self.selected_idx = 0
        else:
            self.selected_idx = new_index
        self._selection_changed()
        self.refresh()

    def scroll_page_forward(self):
        num_lines = self.get_num_lines()
        num_visible_lines = self.get_num_visible_lines()
        new_index = self.selected_idx + num_visible_lines
        if new_index >= num_lines:
            new_index = num_lines - 1
        self.selected_idx = new_index
        self._selection_changed()
        self.refresh()

    def _selection_changed(self):
        num_lines = self.get_num_lines()
        num_visible_lines = self.get_num_visible_lines()
        last_visible_index = self.first_visible_idx + num_visible_lines
        if self.selected_idx >= last_visible_index:
            self.first_visible_idx += (self.selected_idx -
                                       last_visible_index + 1)
        if self.selected_idx < self.first_visible_idx:
            self.first_visible_idx = self.selected_idx
        if self.selected_idx >= 0 and self.selected_idx < len(self.items):
            item = self.items[self.selected_idx]
            item.was_selected()


class Menu(BoxedPanel):

    def __init__(self, title, items):
        max_title_width = 0
        for item in items:
            if max_title_width < len(item.title):
                max_title_width = len(item.title)
        frame = Rect(x=0, y=0, w=max_title_width + 4, h=len(items) + 2)
        super(
            Menu,
            self).__init__(
            frame,
            title=None,
            delegate=None,
            can_become_first_responder=True)
        self.selected_idx = 0
        self.title = title
        self.items = items
        for (item_idx, item) in enumerate(items):
            self.set_line(item_idx, item.title)
        self.hide()

    def update(self):
        super(Menu, self).update()

    def relinquish_first_responder(self):
        if not self.hidden():
            self.hide()

    def perform_action(self):
        selected_idx = self.get_selected_idx()
        if selected_idx < len(self.items):
            action = self.items[selected_idx].action
            if action:
                action()


class MenuBar(Panel):

    def __init__(self, frame):
        super(MenuBar, self).__init__(frame, can_become_first_responder=True)
        self.menus = list()
        self.selected_menu_idx = -1
        self.add_key_action(
            curses.KEY_LEFT,
            self.select_prev,
            "Select the previous menu")
        self.add_key_action(
            curses.KEY_RIGHT,
            self.select_next,
            "Select the next menu")
        self.add_key_action(
            curses.KEY_DOWN,
            lambda: self.select(0),
            "Select the first menu")
        self.add_key_action(
            27,
            self.relinquish_first_responder,
            "Hide current menu")
        self.add_key_action(
            curses.KEY_ENTER,
            self.perform_action,
            "Select the next menu item")
        self.add_key_action(
            10,
            self.perform_action,
            "Select the next menu item")

    def insert_menu(self, menu, index=sys.maxsize):
        if index >= len(self.menus):
            self.menus.append(menu)
        else:
            self.menus.insert(index, menu)
        pt = self.get_position()
        for menu in self.menus:
            menu.set_position(pt)
            pt.x += len(menu.title) + 5

    def perform_action(self):
        '''If no menu is visible, show the first menu. If a menu is visible, perform the action
           associated with the selected menu item in the menu'''
        menu_visible = False
        for menu in self.menus:
            if not menu.hidden():
                menu_visible = True
                break
        if menu_visible:
            menu.perform_action()
            self.selected_menu_idx = -1
            self._selected_menu_changed()
        else:
            self.select(0)

    def relinquish_first_responder(self):
        if self.selected_menu_idx >= 0:
            self.selected_menu_idx = -1
            self._selected_menu_changed()

    def _selected_menu_changed(self):
        for (menu_idx, menu) in enumerate(self.menus):
            is_hidden = menu.hidden()
            if menu_idx != self.selected_menu_idx:
                if not is_hidden:
                    if self.parent.pop_first_responder(menu) == False:
                        menu.hide()
        for (menu_idx, menu) in enumerate(self.menus):
            is_hidden = menu.hidden()
            if menu_idx == self.selected_menu_idx:
                if is_hidden:
                    menu.show()
                    self.parent.push_first_responder(menu)
                menu.top()
        self.parent.refresh()

    def select(self, index):
        if index < len(self.menus):
            self.selected_menu_idx = index
            self._selected_menu_changed()

    def select_next(self):
        num_menus = len(self.menus)
        if self.selected_menu_idx == -1:
            if num_menus > 0:
                self.selected_menu_idx = 0
                self._selected_menu_changed()
        else:
            if self.selected_menu_idx + 1 < num_menus:
                self.selected_menu_idx += 1
            else:
                self.selected_menu_idx = -1
            self._selected_menu_changed()

    def select_prev(self):
        num_menus = len(self.menus)
        if self.selected_menu_idx == -1:
            if num_menus > 0:
                self.selected_menu_idx = num_menus - 1
                self._selected_menu_changed()
        else:
            if self.selected_menu_idx - 1 >= 0:
                self.selected_menu_idx -= 1
            else:
                self.selected_menu_idx = -1
            self._selected_menu_changed()

    def update(self):
        self.erase()
        is_in_first_responder_chain = self.is_in_first_responder_chain()
        if is_in_first_responder_chain:
            self.attron(curses.A_REVERSE)
        pt = Point(x=0, y=0)
        for menu in self.menus:
            self.addstr(pt, '|  ' + menu.title + '  ')
            pt.x += len(menu.title) + 5
        self.addstr(pt, '|')
        width = self.get_size().w
        while pt.x < width:
            self.addch_at_point(pt, ' ')
            pt.x += 1
        if is_in_first_responder_chain:
            self.attroff(curses.A_REVERSE)

        for menu in self.menus:
            menu.update()


class StatusPanel(Panel):

    def __init__(self, frame):
        super(
            StatusPanel,
            self).__init__(
            frame,
            delegate=None,
            can_become_first_responder=False)
        self.status_items = list()
        self.status_dicts = dict()
        self.next_status_x = 1

    def add_status_item(self, name, title, format, width, value, update=True):
        status_item_dict = {'name': name,
                            'title': title,
                            'width': width,
                            'format': format,
                            'value': value,
                            'x': self.next_status_x}
        index = len(self.status_items)
        self.status_items.append(status_item_dict)
        self.status_dicts[name] = index
        self.next_status_x += width + 2
        if update:
            self.update()

    def increment_status(self, name, update=True):
        if name in self.status_dicts:
            status_item_idx = self.status_dicts[name]
            status_item_dict = self.status_items[status_item_idx]
            status_item_dict['value'] = status_item_dict['value'] + 1
        if update:
            self.update()

    def update_status(self, name, value, update=True):
        if name in self.status_dicts:
            status_item_idx = self.status_dicts[name]
            status_item_dict = self.status_items[status_item_idx]
            status_item_dict['value'] = status_item_dict['format'] % (value)
        if update:
            self.update()

    def update(self):
        self.erase()
        for status_item_dict in self.status_items:
            self.addnstr_at_point(
                Point(
                    x=status_item_dict['x'],
                    y=0),
                '%s: %s' %
                (status_item_dict['title'],
                 status_item_dict['value']),
                status_item_dict['width'])

stdscr = None


def intialize_curses():
    global stdscr
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(1)
    try:
        curses.start_color()
    except:
        pass
    return Window(stdscr)


def terminate_curses():
    global stdscr
    if stdscr:
        stdscr.keypad(0)
    curses.echo()
    curses.nocbreak()
    curses.endwin()
